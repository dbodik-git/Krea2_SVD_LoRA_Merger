#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Krea2 SVD LoRA Merger v4: CPU merge, QR + small-core SVD, atomic save."""
import argparse, json, math, os, shutil, sys, tempfile, traceback, contextlib, io, struct, time
import threading
import queue
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from lokr_to_svd_lora import (is_lokr_state_keys, inspect_lokr_header,
                              convert_lokr_state_dict, lokr_alpha_message,
                              lokr_semantics_note, LOKR_ALPHA_STATUS)
from matrix_free_lokr import (LoKrTerm, LoRATerm, MatrixFreeSum,
                              randomized_joint_svd, estimate_workspace_bytes)

# GUI dependencies are optional: the command-line mode works without tkinter.
# The GUI uses ttkbootstrap (pip install ttkbootstrap) and falls back to plain ttk without it.
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    try:
        import ttkbootstrap as ttk
        HAVE_BOOTSTRAP = True
    except ImportError:
        from tkinter import ttk
        HAVE_BOOTSTRAP = False
    HAVE_TK = True
except ImportError:
    tk = ttk = filedialog = messagebox = None
    HAVE_TK = False
    HAVE_BOOTSTRAP = False

THEMES = ("darkly", "superhero", "cyborg", "flatly", "litera", "cosmo")
DEFAULT_THEME = "darkly"


def bs(style):
    """bootstyle kwargs for ttkbootstrap; empty for plain ttk (which would reject the option)."""
    return {"bootstyle": style} if HAVE_BOOTSTRAP else {}


D = Path(r"\krea2")
DEFAULT_INPUTS = [
    D / "lora1.safetensors",
    D / "lora2.safetensors",
    D / "lora3.safetensors",
]
DEFAULT_OUTPUT = D / "Krea2_Merged_3x.safetensors"
A_SUFFIXES = (".lora_A.default.weight", ".lora_down.weight", ".lora_A.weight", ".lora_A.default", ".lora_A")
B_SUFFIXES = (".lora_B.default.weight", ".lora_up.weight", ".lora_B.weight", ".lora_B.default", ".lora_B")
ALPHA_SUFFIXES = (".lora_alpha", ".alpha")


def args_parse():
    p = argparse.ArgumentParser(description="Merge Krea2 LoRAs and SVD-repack to a uniform rank.")
    p.add_argument("--inputs", nargs="+", default=[str(x) for x in DEFAULT_INPUTS])
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="one coefficient per input (default: 1.0 for every input)")
    p.add_argument("--target-rank", type=int, default=32)
    p.add_argument("--cpu-threads", type=int, choices=range(1, 21), default=6,
                   help="PyTorch CPU intra-op threads (default: 6; configurable range: 1-20)")
    p.add_argument("--lokr-full-rank", action="store_true",
                   help="preserve exact LoKr factor rank until the single final merge SVD")
    p.add_argument("--lokr-max-factor-gb", type=float, default=2.0,
                   help="RAM preflight limit for expanded full-rank LoKr A/B factors (default: 2 GB)")
    p.add_argument("--lokr-matrix-free", action="store_true",
                   help="use matrix-free joint SVD for mixed LoKr + LoRA without expanding Kronecker factors")
    p.add_argument("--lokr-svd-oversample", type=int, default=8,
                   help="randomized SVD oversampling (default: 8)")
    p.add_argument("--lokr-svd-power-iterations", type=int, default=2,
                   help="randomized SVD power iterations (default: 2)")
    p.add_argument("--lokr-workspace-max-gb", type=float, default=2.0,
                   help="preflight limit for matrix-free algorithm workspace (default: 2 GB)")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-pause", action="store_true",
                   help="do not wait for Enter after an error (for scripts and .bat files)")
    p.add_argument("--activation-text", default=None,
                   help="activation text for the output Forge Neo JSON "
                        "(default: collected from the input LoRAs' JSON files)")
    p.add_argument("--no-forge-json", action="store_true",
                   help="do not write the Forge Neo sidecar .json next to the output")
    return p.parse_args()


MAX_HEADER_BYTES = 256 * 1024 * 1024


def inspect_lora_header(path):
    """Fast, CPU-only SafeTensors structural check; does not load tensor payloads."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Only .safetensors files are supported.")
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("File too short: no SafeTensors structure.")
        n = struct.unpack("<Q", prefix)[0]
        if n < 2 or n > MAX_HEADER_BYTES or 8 + n > size:
            raise ValueError(f"Invalid SafeTensors header size: {n} bytes.")
        raw = f.read(n)
    try:
        header = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("Invalid SafeTensors header.")

    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("__metadata__ has the wrong format.")

    tensors = {}
    for key, item in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(key, str) or not isinstance(item, dict):
            raise ValueError("Invalid tensor entry in SafeTensors header.")
        shape, dtype = item.get("shape"), item.get("dtype")
        offsets = item.get("data_offsets")
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            raise ValueError(f"Incorrect shape: {key}")
        if not isinstance(dtype, str) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Invalid tensor header entry: {key}")
        if not all(isinstance(x, int) for x in offsets) or offsets[0] < 0 or offsets[1] < offsets[0] or 8 + n + offsets[1] > size:
            raise ValueError(f"Incorrect data_offsets: {key}")
        tensors[key] = {"shape": tuple(shape), "dtype": dtype}

    if is_lokr_state_keys(tensors):
        return inspect_lokr_header(path, metadata, tensors)

    mods, unknown = defaultdict(dict), []
    for key, info in tensors.items():
        parsed = parse_key(key)
        if parsed is None:
            unknown.append(key)
            continue
        kind, stem, suffix = parsed
        if kind in mods[stem]:
            raise ValueError(f"Duplicate {kind} for layer {stem}.")
        mods[stem][kind] = info
        mods[stem][kind + "_suffix"] = suffix

    errors = []
    if unknown:
        errors.append("Unrecognized tensors: " + ", ".join(unknown[:8]))
    if not mods:
        errors.append("No LoRA A/B tensors found.")
    ranks, dtypes = set(), set()
    for stem, m in mods.items():
        if "A" not in m or "B" not in m:
            errors.append(f"Incomplete A/B pair: {stem}")
            continue
        ash, bsh = m["A"]["shape"], m["B"]["shape"]
        if len(ash) != 2 or len(bsh) != 2 or ash[0] != bsh[1] or ash[0] < 1:
            errors.append(f"Wrong A/B shapes in {stem}: A={ash}, B={bsh}")
            continue
        ranks.add(ash[0]); dtypes.update((m["A"]["dtype"], m["B"]["dtype"]))
        if "alpha" in m:
            alpha_shape = m["alpha"]["shape"]
            alpha_numel = math.prod(alpha_shape) if alpha_shape else 1
            if len(alpha_shape) > 1 or alpha_numel != 1:
                errors.append(f"Invalid alpha shape in layer {stem}: {alpha_shape} (expected scalar or one-element tensor)")

    if errors:
        raise ValueError("\n".join(errors))

    # "warnings" need a user decision; "notes" are informational only.
    warnings, notes = [], []
    if not metadata.get("modelspec.architecture"):
        notes.append("Metadata has no modelspec.architecture.")
    if not (metadata.get("ss_base_model_version") or metadata.get("ss_base_model")):
        notes.append("Metadata has no base model information.")
    if len(ranks) > 1:
        notes.append("Layers use different ranks: " + ", ".join(map(str, sorted(ranks))))
    return {"path": path, "metadata": metadata, "modules": mods,
            "tensor_count": len(tensors), "module_count": len(mods),
            "ranks": sorted(ranks), "dtypes": sorted(dtypes),
            "warnings": warnings, "notes": notes}


def compare_lora_headers(new_info, existing_infos):
    """Check shared target dimensions against already selected LoRAs."""
    warnings = []
    for old in existing_infos:
        shared = set(new_info["modules"]) & set(old["modules"])
        if not shared:
            warnings.append(f"No shared layers with {old['path'].name}.")
            continue
        bad = []
        for stem in shared:
            a, b = new_info["modules"][stem], old["modules"][stem]
            if a["A"]["shape"][1] != b["A"]["shape"][1] or a["B"]["shape"][0] != b["B"]["shape"][0]:
                bad.append(stem)
        if bad:
            raise ValueError(
                f"Dimension mismatch with {old['path'].name}: {len(bad)} shared layers.\n" +
                "First mismatching layers: " + ", ".join(bad[:6])
            )
        ma, mb = new_info["metadata"], old["metadata"]
        aa, ab = ma.get("modelspec.architecture"), mb.get("modelspec.architecture")
        ba = ma.get("ss_base_model_version") or ma.get("ss_base_model")
        bb = mb.get("ss_base_model_version") or mb.get("ss_base_model")
        # "krea2/lokr" vs "krea2/lora" is the same model family, not a real mismatch.
        if aa and ab and str(aa).lower().replace("lokr", "lora") != str(ab).lower().replace("lokr", "lora"):
            warnings.append(f"Architecture differs from {old['path'].name}: {aa} / {ab}.")
        if ba and bb and ba != bb:
            warnings.append(f"Base model differs from {old['path'].name}: {ba} / {bb}.")
    return warnings


def parse_key(key):
    for kind, suffixes in (("A", A_SUFFIXES), ("B", B_SUFFIXES), ("alpha", ALPHA_SUFFIXES)):
        for s in suffixes:
            if key.endswith(s):
                return kind, key[:-len(s)], s
    return None


def load_lora(path, torch, load_file, target_rank, lokr_full_rank=False, metadata=None,
              stage1_report=None):
    raw = load_file(str(path), device="cpu")
    from_lokr = is_lokr_state_keys(raw)
    if from_lokr:
        raw = convert_lokr_state_dict(
            raw, torch, None if lokr_full_rank else target_rank,
            source_name=path.name, metadata=metadata, stage1_report=stage1_report
        )
    mods, unknown = defaultdict(dict), []
    for key, tensor in raw.items():
        parsed = parse_key(key)
        if parsed is None:
            unknown.append(key); continue
        kind, stem, suffix = parsed
        if kind in mods[stem]:
            raise ValueError(f"{path.name}: duplicate {kind} for {stem}")
        mods[stem][kind] = tensor.contiguous()
        mods[stem][kind + "_suffix"] = suffix
    if unknown:
        raise ValueError(f"{path.name}: unrecognized tensor keys (stopped, not dropped): " + ", ".join(unknown[:8]))
    for stem, m in mods.items():
        if "A" not in m or "B" not in m:
            raise ValueError(f"{path.name}: incomplete A/B pair: {stem}")
        A, B = m["A"], m["B"]
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
            raise ValueError(f"{path.name}: invalid factor shapes for {stem}: A={tuple(A.shape)} B={tuple(B.shape)}")
        if "alpha" in m:
            a = m["alpha"].float().reshape(-1)
            if a.numel() != 1: raise ValueError(f"{path.name}: alpha must be scalar: {stem}")
            m["alpha_value"] = float(a[0].item())
        else:
            m["alpha_value"] = float(A.shape[0])
        m["rank"] = int(A.shape[0])
        m["from_lokr"] = from_lokr
    return mods


def repack(A_cat, B_cat, target_rank, torch):
    # B=Qb Rb and A.T=Qa Ra, so BA=Qb (Rb Ra.T) Qa.T.
    A, B = A_cat.float(), B_cat.float()
    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")
    core = Rb @ Ra.T
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)
    k = min(target_rank, int(S.numel()))
    root = torch.sqrt(torch.clamp(S[:k], min=0))
    Bk = (Qb @ U[:, :k]) * root.unsqueeze(0)
    Ak = root.unsqueeze(1) * (Vh[:k, :] @ Qa.T)
    Bout = B.new_zeros((B.shape[0], target_rank))
    Aout = A.new_zeros((target_rank, A.shape[1]))
    if k:
        Bout[:, :k] = Bk
        Aout[:k, :] = Ak
    norm = torch.linalg.vector_norm(S)
    err = 0.0 if float(norm.item()) == 0 else float((torch.linalg.vector_norm(S[k:]) / norm).item())
    return Aout, Bout, err, int(S.numel())


def repack_ex(A_cat, B_cat, target_rank, torch):
    """Like repack() but also returns norms: (A, B, rel_err, core_rank, total_norm, tail_norm).

    rel_err is relative to the *input* of the repack (the already-truncated sum when
    LoKr was converted first), so it does not include earlier truncation stages.
    """
    A, B = A_cat.float(), B_cat.float()
    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    k = min(target_rank, int(S.numel()))
    root = torch.sqrt(torch.clamp(S[:k], min=0))
    Bout = B.new_zeros((B.shape[0], target_rank))
    Aout = A.new_zeros((target_rank, A.shape[1]))
    if k:
        Bout[:, :k] = (Qb @ U[:, :k]) * root.unsqueeze(0)
        Aout[:k, :] = root.unsqueeze(1) * (Vh[:k, :] @ Qa.T)
    total = float(torch.linalg.vector_norm(S.double()).item())
    tail = float(torch.linalg.vector_norm(S[k:].double()).item()) if k < S.numel() else 0.0
    return Aout, Bout, (0.0 if total == 0 else tail / total), int(S.numel()), total, tail


def combined_error_bound(stage1_abs_errors, weights, total_norm, tail_norm):
    """Rigorous upper bound on ||M - final||_F / ||M||_F, or None if not provable.

    M = sum_i w_i M_i, P = sum_i w_i P_i (P_i = truncated LoKr / exact LoRA), final = T(P).
    ||M - final|| <= ||M - P|| + ||P - final|| <= sum_i |w_i| e_i + tail, and
    ||M|| >= ||P|| - sum_i |w_i| e_i. Triangle inequality only: usually loose.
    """
    e1 = sum(abs(float(w)) * float(e) for w, e in zip(weights, stage1_abs_errors))
    denom = total_norm - e1
    if denom <= 0:
        return None
    return (e1 + tail_norm) / denom


def _pick_global_output_style(candidates):
    """Choose output key suffixes ONCE for the whole file, independent of input order.

    candidates: iterable of (is_lokr, module_dict) over ALL modules of ALL inputs.
    Ordinary-LoRA key styles are ranked by module count (desc), then lexicographically,
    so permuting the input files cannot change the result and no file ends up with
    mixed key styles. Without any ordinary LoRA, kohya names are used.
    Returns (a_suffix, b_suffix, alpha_suffix_or_None, note_or_None).
    """
    counts, alphas = Counter(), defaultdict(Counter)
    for is_lokr, mod in candidates:
        if is_lokr:
            continue
        style = (mod["A_suffix"], mod["B_suffix"])
        counts[style] += 1
        if mod.get("alpha_suffix"):
            alphas[style][mod["alpha_suffix"]] += 1
    if not counts:
        return ".lora_down.weight", ".lora_up.weight", ".alpha", None
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    style = ranked[0][0]
    alpha = None
    if alphas.get(style):
        alpha = sorted(alphas[style].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    note = None
    if len(ranked) > 1:
        note = ("NOTE: inputs use different key styles "
                + "; ".join(f"{a}/{b} x{c}" for (a, b), c in ranked)
                + f". Writing ONE style for every module: {style[0]} / {style[1]}.")
    return style[0], style[1], alpha, note


def _stem_seed(stem, base=1729):
    """Per-module RNG seed that depends only on the module name (stable across runs,
    input order and the presence of other modules)."""
    return (base + zlib.crc32(stem.encode("utf-8"))) & 0x7FFFFFFF


def _pick_output_style(candidates):
    """Choose output key suffixes once, the same way for every module.

    candidates: iterable of (is_lokr, module_dict). The first ordinary LoRA wins, so a
    PEFT-style (lora_A/lora_B) input is never silently turned into kohya-style just
    because a LoKr file was listed first. With only LoKr inputs, kohya names are used.
    Returns (a_suffix, b_suffix, alpha_suffix_or_None).
    """
    candidates = list(candidates)
    ordinary = next((mod for is_lokr, mod in candidates if not is_lokr), None)
    if ordinary is None:
        return ".lora_down.weight", ".lora_up.weight", ".alpha"
    return ordinary["A_suffix"], ordinary["B_suffix"], ordinary.get("alpha_suffix")


def _output_identity_metadata(input_metadatas):
    """modelspec.architecture / ss_base_model_version for the output.

    Taken from the inputs when they all agree; otherwise the Krea2 defaults are used.
    An architecture string that mentions LoKr is not copied to an ordinary LoRA output.
    """
    defaults = {"modelspec.architecture": "krea2/lora", "ss_base_model_version": "krea2"}
    arch = {md.get("modelspec.architecture") for md in input_metadatas if md.get("modelspec.architecture")}
    base = {md.get("ss_base_model_version") or md.get("ss_base_model")
            for md in input_metadatas if md.get("ss_base_model_version") or md.get("ss_base_model")}
    result = dict(defaults)
    if len(arch) == 1:
        value = next(iter(arch))
        if "lokr" not in str(value).lower():
            result["modelspec.architecture"] = str(value)
    elif len(arch) > 1:
        print(f"NOTE: inputs disagree on modelspec.architecture {sorted(map(str, arch))}; "
              f"writing default {defaults['modelspec.architecture']}.")
    if len(base) == 1:
        result["ss_base_model_version"] = str(next(iter(base)))
    elif len(base) > 1:
        print(f"NOTE: inputs disagree on base model {sorted(map(str, base))}; "
              f"writing default {defaults['ss_base_model_version']}.")
    return result


def _warn_alpha_metadata(name, no_alpha_ranks, metadata):
    """Warn if alpha-less modules assume scale 1 but metadata says otherwise."""
    meta_alpha = (metadata or {}).get("ss_network_alpha")
    if not no_alpha_ranks or meta_alpha is None:
        return
    try:
        meta_alpha = float(meta_alpha)
    except (TypeError, ValueError):
        return
    if any(abs(meta_alpha - r) > 1e-9 for r in no_alpha_ranks):
        print(f"  WARNING: {name} has no alpha tensors, so scale 1 (alpha = rank) is assumed, "
              f"but metadata says ss_network_alpha={meta_alpha:g}. The contribution may be scaled incorrectly.")


FORGE_USER_METADATA_KEYS = (
    "description", "sd version", "activation text",
    "preferred weight", "negative text", "notes",
)
# Used only when none of the source JSON files specifies an "sd version".
FORGE_SD_VERSION_FALLBACK = "krea"


def _read_forge_user_metadata(lora_path):
    """Read Forge Neo's sidecar JSON; return an empty mapping when absent."""
    json_path = Path(lora_path).with_suffix(".json")
    if not json_path.is_file():
        return {}
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  WARNING: Could not read Forge metadata {json_path.name}: {exc}")
        return {}


def _merge_text_fields(values):
    """Join comma-separated metadata phrases, preserving order and removing duplicates."""
    result, seen = [], set()
    for value in values:
        if not isinstance(value, str):
            continue
        for phrase in value.split(","):
            phrase = phrase.strip()
            key = phrase.casefold()
            if phrase and key not in seen:
                seen.add(key)
                result.append(phrase)
    return ", ".join(result)


def build_forge_user_metadata(input_paths, activation_text=None):
    """Build Forge Neo's six-key user metadata from input LoRA sidecar JSON files."""
    sources = [_read_forge_user_metadata(path) for path in input_paths]
    names = [Path(path).stem for path in input_paths]
    notes = [item.get("notes", "").strip() for item in sources
             if isinstance(item.get("notes"), str) and item.get("notes").strip()]
    # In Forge a preferred weight of 0 means "not set", so only real values take part in the average.
    weights = []
    for item in sources:
        try:
            weight = float(item.get("preferred weight", 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(weight) and weight > 0:
            weights.append(weight)
    preferred_weight = sum(weights) / len(weights) if weights else 0.0
    versions = [v.strip() for v in (item.get("sd version") for item in sources)
                if isinstance(v, str) and v.strip() and v.strip().lower() != "unknown"]
    sd_version = (sorted(Counter(versions).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                  if versions else FORGE_SD_VERSION_FALLBACK)
    metadata = {
        "description": "Merged LoRA: " + ", ".join(names),
        "sd version": sd_version,
        "activation text": (activation_text if activation_text is not None else
                           _merge_text_fields([item.get("activation text", "") for item in sources])),
        "preferred weight": preferred_weight,
        "negative text": _merge_text_fields([item.get("negative text", "") for item in sources]),
        "notes": "Merged from: " + ", ".join(names)
                 + (("\nSource notes:\n" + "\n".join(notes)) if notes else ""),
    }
    return {key: metadata[key] for key in FORGE_USER_METADATA_KEYS}


def write_forge_user_metadata(output_path, metadata, overwrite=False):
    """Atomically write Forge Neo-compatible sidecar JSON beside the merged LoRA."""
    json_path = Path(output_path).with_suffix(".json")
    if json_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Forge metadata JSON already exists: {json_path}. "
                "Enable overwrite or choose another output name."
            )
        # Keep the previous sidecar (it may hold hand-written notes).
        backup = json_path.with_name(json_path.name + ".bak")
        try:
            shutil.copy2(str(json_path), str(backup))
            print(f"Previous Forge metadata JSON backed up to: {backup.name}")
        except OSError as exc:
            print(f"  WARNING: Could not back up the existing JSON: {exc}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        fd, name = tempfile.mkstemp(prefix=json_path.name + ".", suffix=".tmp",
                                    dir=str(json_path.parent))
        os.close(fd)
        temp = Path(name)
        with temp.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)
            file.write("\n")
        os.replace(str(temp), str(json_path))
        temp = None
    finally:
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
    print(f"Forge Neo user metadata JSON: {json_path}")




def _process_working_set_bytes():
    """Return (current working set, OS lifetime peak working set) in bytes."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
            psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return (
                    int(counters.WorkingSetSize),
                    int(counters.PeakWorkingSetSize),
                )
        except Exception:
            pass

    # Optional dependency fallback; does not affect the merge if unavailable.
    try:
        import os
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        peak = getattr(info, "peak_wset", None)
        return int(info.rss), (int(peak) if peak else None)
    except Exception:
        pass

    # Linux fallback without requiring psutil.
    if sys.platform.startswith("linux"):
        try:
            import os
            import resource

            with open("/proc/self/statm", "r", encoding="ascii") as file:
                resident_pages = int(file.read().split()[1])
            current = resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            return current, peak
        except Exception:
            pass
    return None, None


class _ProcessMemoryMonitor:
    """Sample process RSS during one merge; samples do not retain tensor payloads."""
    def __init__(self, interval=0.05):
        self.interval = float(interval)
        self.start_rss = None
        self.current_rss = None
        self.peak_sampled_rss = None
        self.peak_os_working_set = None
        self._stop_event = threading.Event()
        self._thread = None

    def _sample(self):
        rss, os_peak = _process_working_set_bytes()
        if rss is not None:
            self.current_rss = rss
            if self.peak_sampled_rss is None or rss > self.peak_sampled_rss:
                self.peak_sampled_rss = rss
        if os_peak is not None:
            if self.peak_os_working_set is None or os_peak > self.peak_os_working_set:
                self.peak_os_working_set = os_peak

    def _run(self):
        while not self._stop_event.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self.start_rss = self.current_rss
        self._thread = threading.Thread(
            target=self._run, name="MergerMemoryMonitor", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def _print_run_diagnostics(status, elapsed_seconds, cpu_threads, monitor):
    """Print end-of-run timing and process-RSS diagnostics."""
    print()
    print("Run diagnostics")
    print(f"  Result: {status}")
    print(f"  Elapsed wall time: {elapsed_seconds:.2f} s")
    print(f"  PyTorch CPU intra-op threads: {cpu_threads}")
    mib = 1024 ** 2
    if monitor.start_rss is not None:
        print(f"  Process RSS at start: {monitor.start_rss / mib:.1f} MiB")
    else:
        print("  Process RSS at start: unavailable")
    if monitor.current_rss is not None:
        print(f"  Process RSS at end: {monitor.current_rss / mib:.1f} MiB")
    else:
        print("  Process RSS at end: unavailable")
    if monitor.peak_sampled_rss is not None:
        peak_mib = monitor.peak_sampled_rss / mib
        print(f"  Peak sampled process working set: {peak_mib:.1f} MiB")
        if monitor.start_rss is not None:
            growth = max(0, monitor.peak_sampled_rss - monitor.start_rss)
            print(f"  Peak working-set increase over run start: {growth / mib:.1f} MiB")
    else:
        print("  Peak sampled process working set: unavailable")
    if monitor.peak_os_working_set is not None:
        print(f"  OS-reported lifetime peak working set: "
              f"{monitor.peak_os_working_set / mib:.1f} MiB")
    else:
        print("  OS-reported lifetime peak working set: unavailable")
    print("  Working set includes shared and private resident pages, including mapped-file pages.")
    print("  This is process memory, not total system RAM; sampled peak uses a 50 ms interval.")


def merge_loras_matrix_free(args, torch, safe_open, load_file, save_file,
                            inputs, output, write_json, output_json, check_cancel):
    """Merge one module at a time through an implicit LoKr/LoRA operator."""
    if bool(getattr(args, "lokr_full_rank", False)):
        raise ValueError("--lokr-matrix-free and --lokr-full-rank are separate modes; choose only one.")
    oversample = int(getattr(args, "lokr_svd_oversample", 8))
    power_iterations = int(getattr(args, "lokr_svd_power_iterations", 2))
    workspace_limit_gb = float(getattr(args, "lokr_workspace_max_gb", 2.0))
    if oversample < 0 or power_iterations < 0:
        raise ValueError("Matrix-free oversample and power iterations must be non-negative.")
    if not math.isfinite(workspace_limit_gb) or workspace_limit_gb <= 0:
        raise ValueError("--lokr-workspace-max-gb must be finite and greater than zero.")

    infos = [inspect_lora_header(p) for p in inputs]
    for path, info in zip(inputs, infos):
        if info.get("format") == "lokr":
            continue
        no_alpha_ranks = {
            int(mod["A"]["shape"][0]) for mod in info["modules"].values()
            if "alpha_suffix" not in mod
        }
        metadata_alpha = info.get("metadata", {}).get("ss_network_alpha")
        if no_alpha_ranks and metadata_alpha is not None:
            try:
                metadata_alpha = float(metadata_alpha)
            except (TypeError, ValueError):
                metadata_alpha = None
            if metadata_alpha is not None and any(
                abs(metadata_alpha - rank) > 1e-9 for rank in no_alpha_ranks
            ):
                print(
                    f"WARNING: {path.name} has LoRA modules without alpha tensors; "
                    f"scale 1 (alpha = rank) is assumed, but metadata says "
                    f"ss_network_alpha={metadata_alpha:g}."
                )
    for path, info in zip(inputs, infos):
        if info.get("format") != "lokr":
            continue
        alpha_stems = [stem for stem, mod in info["modules"].items() if mod.get("has_alpha")]
        print(lokr_semantics_note(path.name, len(alpha_stems), len(info["modules"])))
        if not alpha_stems:
            continue
        alpha_values = []
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for stem in alpha_stems:  # scalars only: negligible memory
                tensor = handle.get_tensor(stem + ".alpha").float().reshape(-1)
                if tensor.numel() == 1 and torch.isfinite(tensor).all():
                    alpha_values.append(float(tensor[0].item()))
        message = lokr_alpha_message(path.name, alpha_values, info.get("metadata"))
        if message:
            print(message)
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            compare_lora_headers(infos[i], [infos[j]])
    stems = sorted(set().union(*(set(info["modules"]) for info in infos)))
    if not stems:
        raise ValueError("No compatible LoRA modules were found.")

    max_workspace = 0
    max_workspace_stem = None
    for stem in stems:
        first = next(info["modules"][stem] for info in infos if stem in info["modules"])
        shape = (first["B"]["shape"][0], first["A"]["shape"][1])
        required = estimate_workspace_bytes(
            shape, args.target_rank, oversample=oversample, safety_factor=2.5
        )
        if required > max_workspace:
            max_workspace, max_workspace_stem = required, stem
    limit_bytes = workspace_limit_gb * (1024 ** 3)
    print("Matrix-free LoKr/LoRA joint SVD mode")
    print(f"Workspace preflight: largest estimated algorithm workspace "
          f"{max_workspace / (1024 ** 3):.3f} GiB at {max_workspace_stem}; "
          f"configured limit {workspace_limit_gb:.2f} GiB.")
    print("Estimate excludes source tensor payloads, output tensors and Python/PyTorch runtime.")
    if max_workspace > limit_bytes:
        raise MemoryError(
            "Matrix-free SVD stopped before loading tensor payloads: estimated workspace "
            f"{max_workspace / (1024 ** 3):.3f} GiB exceeds the configured "
            f"{workspace_limit_gb:.2f} GiB limit."
        )

    activation_override = getattr(args, "activation_text", None)
    forge_metadata = (build_forge_user_metadata(inputs, activation_text=activation_override)
                      if write_json else None)
    g_a, g_b, g_alpha, style_note = _pick_global_output_style(
        (info.get("format") == "lokr", mod) for info in infos for mod in info["modules"].values())
    if style_note:
        print(style_note)
    if args.dry_run:
        print(f"DRY RUN: validated {len(inputs)} inputs and {len(stems)} modules (headers only).")
        print(f"Planned merge: matrix-free randomized joint SVD, target rank {args.target_rank}, "
              f"oversample {oversample}, power iterations {power_iterations}, dtype {args.dtype}; "
              f"output keys {g_a} / {g_b}" + (f" / {g_alpha}" if g_alpha else "") + ".")
        print("Header checks do not prove the merge will succeed or fit in RAM.")
        if forge_metadata is not None:
            print(f"Forge Neo JSON that would be written: {output_json}")
            print(json.dumps(forge_metadata, indent=4, ensure_ascii=False))
        print("No tensor payloads loaded (only scalar LoKr alpha values were read); no output written.")
        return

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if args.dtype not in dtypes:
        raise ValueError(f"Unsupported output dtype: {args.dtype}")
    output_dtype = dtypes[args.dtype]
    out, errors, ranks = {}, [], set()
    # Keep file handles open, but only load tensors for the current module.
    with contextlib.ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for path in inputs
        ]
        for index, stem in enumerate(stems, 1):
            check_cancel()
            lokr_terms, lora_terms = [], []
            present_infos = []
            for file_index, (info, handle, weight) in enumerate(zip(infos, handles, args.weights)):
                mod = info["modules"].get(stem)
                if mod is None:
                    continue
                present_infos.append((info, mod))
                if info.get("format") == "lokr":
                    w1 = handle.get_tensor(stem + ".lokr_w1")
                    w2 = handle.get_tensor(stem + ".lokr_w2")
                    lokr_terms.append(LoKrTerm(w1, w2, weight=float(weight)))
                else:
                    A = handle.get_tensor(stem + mod["A_suffix"])
                    B = handle.get_tensor(stem + mod["B_suffix"])
                    rank = int(A.shape[0])
                    if "alpha_suffix" in mod:
                        alpha_tensor = handle.get_tensor(stem + mod["alpha_suffix"]).float().reshape(-1)
                        if alpha_tensor.numel() != 1 or not torch.isfinite(alpha_tensor).all():
                            raise ValueError(f"Invalid alpha tensor at {stem} in {inputs[file_index].name}.")
                        alpha = float(alpha_tensor[0].item())
                    else:
                        alpha = float(rank)
                    lora_terms.append(LoRATerm(A, B, scale=alpha / rank, weight=float(weight)))

            operator = MatrixFreeSum(torch, lokr_terms=lokr_terms, lora_terms=lora_terms)
            Aout, Bout, error_estimate, core_rank = randomized_joint_svd(
                operator, args.target_rank, oversample=oversample,
                power_iterations=power_iterations, seed=_stem_seed(stem),
                error_probes=8,
            )
            if not torch.isfinite(Aout).all() or not torch.isfinite(Bout).all():
                raise FloatingPointError(f"Non-finite matrix-free factor at {stem}.")
            Aout = Aout.to(output_dtype).contiguous()
            Bout = Bout.to(output_dtype).contiguous()
            if not torch.isfinite(Aout.float()).all() or not torch.isfinite(Bout.float()).all():
                raise FloatingPointError(f"{args.dtype} overflow at {stem}; try bf16 or fp32.")

            a_suffix, b_suffix, alpha_suffix = g_a, g_b, g_alpha
            a_key, b_key = stem + a_suffix, stem + b_suffix
            if a_key in out or b_key in out:
                raise ValueError(f"Output key collision at {stem}.")
            out[a_key], out[b_key] = Aout, Bout
            if alpha_suffix:
                out[stem + alpha_suffix] = torch.tensor(float(args.target_rank), dtype=torch.float32)
            ranks.add(int(Aout.shape[0]))
            errors.append((error_estimate, stem, core_rank))
            if index % 10 == 0 or index == len(stems):
                print(f"  matrix-free modules {index}/{len(stems)}")

    mean_error = sum(item[0] for item in errors) / max(1, len(errors))
    max_error = max((item[0] for item in errors), default=0.0)
    print(f"Random-probe relative residual estimate: mean={mean_error:.6g}; max={max_error:.6g}")
    print("This is a randomized estimate, not an exact Frobenius truncation error.")
    for error, stem, core_rank in sorted(errors, reverse=True)[:10]:
        print(f"  estimated residual {error:.6g} (projected rank {core_rank}) {stem}")

    metadata = {
        "format": "pt", **_output_identity_metadata([info["metadata"] for info in infos]),
        "ss_network_dim": str(args.target_rank),
        "ss_network_alpha": str(args.target_rank), "ss_output_name": output.stem,
        "modelspec.title": output.stem, "merge_method": "matrix_free_randomized_joint_svd",
        "merge_input_files": json.dumps([p.name for p in inputs], ensure_ascii=False),
        "merge_weights": json.dumps(args.weights), "merge_target_rank": str(args.target_rank),
        "merge_output_dtype": args.dtype,
        "merge_error_metric": "random_probe_relative_residual_estimate",
        **({"merge_lokr_alpha_semantics": LOKR_ALPHA_STATUS}
           if any(i.get("format") == "lokr" for i in infos) else {}),
        "merge_svd_oversample": str(oversample),
        "merge_svd_power_iterations": str(power_iterations),
    }
    check_cancel()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        fd, name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp.safetensors",
                                    dir=str(output.parent))
        os.close(fd)
        temp = Path(name)
        save_file(out, str(temp), metadata=metadata)
        check = load_file(str(temp), device="cpu")
        if set(check) != set(out):
            raise RuntimeError("Serialized keys differ from in-memory keys.")
        for key, tensor in check.items():
            if tuple(tensor.shape) != tuple(out[key].shape):
                raise RuntimeError(f"Serialized shape mismatch: {key}")
            if not torch.isfinite(tensor.float()).all():
                raise RuntimeError(f"Non-finite serialized tensor: {key}")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output appeared during processing: {output}")
        os.replace(str(temp), str(output))
        temp = None
    finally:
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
    print(f"SUCCESS: {output}")
    print(f"Output tensors: {len(out)}; factor ranks: {sorted(ranks)}")
    print("Saved file was reopened and validated before atomic replacement.")
    if forge_metadata is None:
        print("Forge Neo JSON skipped (--no-forge-json).")
    else:
        try:
            write_forge_user_metadata(output, forge_metadata, overwrite=args.overwrite)
        except Exception as exc:
            print(f"WARNING: LoRA saved, but Forge Neo JSON could not be written: {exc}")

def _merge_loras_impl(args=None):
    if args is None:
        args = args_parse()
    if not args.inputs: raise ValueError("At least one input is required.")
    if getattr(args, "weights", None) is None:
        args.weights = [1.0] * len(args.inputs)
    if len(args.weights) != len(args.inputs): raise ValueError("Supply exactly one --weights value per input.")
    if args.target_rank < 1: raise ValueError("--target-rank must be >= 1.")
    lokr_max_factor_gb = float(getattr(args, "lokr_max_factor_gb", 2.0))
    if not math.isfinite(lokr_max_factor_gb) or lokr_max_factor_gb <= 0:
        raise ValueError("--lokr-max-factor-gb must be a finite value greater than zero.")
    if not all(math.isfinite(w) for w in args.weights): raise ValueError("Weights must be finite.")
    import torch
    cpu_threads = int(getattr(args, "cpu_threads", 6))
    if not 1 <= cpu_threads <= 20:
        raise ValueError("--cpu-threads must be between 1 and 20.")
    # Match the benchmarked PyTorch CPU intra-op setting. Do this before
    # loading tensor payloads or running any eager tensor operations.
    torch.set_num_threads(cpu_threads)
    print(f"PyTorch CPU intra-op threads: {torch.get_num_threads()}")
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    inputs = [Path(x).expanduser() for x in args.inputs]
    output = Path(args.output).expanduser()
    if output.suffix.lower() != ".safetensors":
        raise ValueError("Output file must have the .safetensors extension.")
    cancel_event = getattr(args, "cancel_event", None)

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Merge cancelled by user. No output file was written.")

    if output.resolve() in {p.resolve() for p in inputs}: raise ValueError("Output cannot overwrite an input.")

    missing = [str(p) for p in inputs if not p.is_file()]
    if missing: raise FileNotFoundError("Missing input(s):\n  " + "\n  ".join(missing))
    # A dry run never writes anything, so an existing output file is not a problem there.
    if output.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output exists: {output}. Choose another --output or pass --overwrite.")
    write_json = not getattr(args, "no_forge_json", False)
    output_json = output.with_suffix(".json")
    if write_json:
        if output_json.resolve() in {p.with_suffix(".json").resolve() for p in inputs}:
            raise ValueError("The output Forge JSON would overwrite an input LoRA's JSON file; choose another output name.")
        if output_json.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"Forge metadata JSON exists: {output_json}. Choose another output or pass --overwrite.")

    if bool(getattr(args, "lokr_matrix_free", False)):
        if bool(getattr(args, "lokr_full_rank", False)):
            raise ValueError("--lokr-matrix-free and --lokr-full-rank are mutually exclusive.")
        print("Krea2 SVD LoRA Merger")
        print(f"Target rank={args.target_rank}; dtype={args.dtype}; sources remain read-only.")
        for p, w in zip(inputs, args.weights):
            print(f"  {p.name}  coefficient={w:g}")
        print(f"Output: {output}\n")
        merge_loras_matrix_free(
            args, torch, safe_open, load_file, save_file, inputs, output,
            write_json, output_json, check_cancel
        )
        return
    print("Krea2 SVD LoRA Merger")
    print(f"Target rank={args.target_rank}; dtype={args.dtype}; sources remain read-only.")
    lokr_full_rank = bool(getattr(args, "lokr_full_rank", False))
    if lokr_full_rank:
        infos = [inspect_lora_header(p) for p in inputs]
        estimated = sum(info.get("lokr_full_rank_factor_bytes", 0) for info in infos)
        limit_bytes = lokr_max_factor_gb * (1024 ** 3)
        print(f"LoKr exact-factor preflight: estimated A/B storage={estimated / (1024 ** 3):.2f} GiB; "
              f"configured limit={limit_bytes / (1024 ** 3):.2f} GiB.")
        if estimated > limit_bytes:
            raise MemoryError(
                "Full-rank LoKr expansion was stopped before loading tensor payloads: "
                f"estimated A/B factors need {estimated / (1024 ** 3):.2f} GiB, "
                f"above the configured {limit_bytes / (1024 ** 3):.2f} GiB limit. "
                "Increase --lokr-max-factor-gb only if sufficient physical RAM is available."
            )
        print("LoKr mode: preserve full factor rank; only the final joint SVD will truncate.")
    for p, w in zip(inputs, args.weights): print(f"  {p.name}  coefficient={w:g}")
    print(f"Output: {output}\n")
    if args.dry_run:
        # Header-only validation: no tensor payloads are read and no LoKr is converted.
        infos = [inspect_lora_header(p) for p in inputs]
        for p, info in zip(inputs, infos):
            check_cancel()
            md = info["metadata"]
            print(f"Checked {p.name}: {info['module_count']} modules; format={info.get('format', 'lora')}; "
                  f"architecture={md.get('modelspec.architecture', 'unspecified')}; "
                  f"base={md.get('ss_base_model_version', 'unspecified')}")
            if info.get("format") == "lokr":
                print(lokr_semantics_note(p.name, sum(1 for m_ in info["modules"].values() if m_.get("has_alpha")),
                                          len(info["modules"])))
            if info.get("format") != "lokr":
                _warn_alpha_metadata(
                    p.name,
                    {int(m["A"]["shape"][0]) for m in info["modules"].values() if "alpha_suffix" not in m},
                    md)
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                for warning in compare_lora_headers(infos[j], [infos[i]]):
                    print(f"  WARNING: {warning}")
                left, right = set(infos[i]["modules"]), set(infos[j]["modules"])
                print(f"Overlap {inputs[i].name} <-> {inputs[j].name}: {len(left & right)} shared; "
                      f"{len(left - right)} left-only; {len(right - left)} right-only")
        union = set().union(*(set(info["modules"]) for info in infos))
        print(f"\nUnion: {len(union)} module stems; all detected shapes compatible.")
        activation_override = getattr(args, "activation_text", None)
        forge_metadata = (build_forge_user_metadata(inputs, activation_text=activation_override)
                          if write_json else None)
        if forge_metadata is not None:
            print(f"\nForge Neo JSON that would be written: {output_json}")
            print(json.dumps(forge_metadata, indent=4, ensure_ascii=False))
        g_a, g_b, g_alpha, style_note = _pick_global_output_style(
            (info.get("format") == "lokr", mod) for info in infos for mod in info["modules"].values())
        if style_note:
            print(style_note)
        mode = ("exact LoKr rank kept until the final SVD (--lokr-full-rank)" if lokr_full_rank else
                f"LoKr -> rank {args.target_rank} -> final rank {args.target_rank} (two truncations; "
                "see the combined-error line at merge time)")
        print(f"Planned merge: dense QR + small-core SVD, target rank {args.target_rank}, dtype {args.dtype}; "
              f"LoKr handling: {mode}; output keys {g_a} / {g_b}" + (f" / {g_alpha}" if g_alpha else "") + ".")
        print("DRY RUN complete (headers only; no tensor payloads loaded). This does NOT prove the merge "
              "will succeed or fit in RAM. No output written.")
        return
    allmods = []
    input_metadata = []
    stage1_reports = {}   # file index -> {stem: absolute LoKr truncation error}
    for p in inputs:
        check_cancel()
        with safe_open(str(p), framework="pt", device="cpu") as sf: md = sf.metadata() or {}
        input_metadata.append(md)
        mods = load_lora(p, torch, load_file, args.target_rank, lokr_full_rank=lokr_full_rank, metadata=md,
                         stage1_report=stage1_reports.setdefault(len(allmods), {}))
        allmods.append(mods)
        print(f"Read {p.name}: {len(mods)} modules; architecture={md.get('modelspec.architecture','unspecified')}; base={md.get('ss_base_model_version','unspecified')}")
        # Without alpha tensors the scale is assumed to be 1 (alpha = rank). Warn if metadata disagrees.
        no_alpha_ranks = {m["rank"] for m in mods.values() if "alpha" not in m}
        _warn_alpha_metadata(p.name, no_alpha_ranks, md)
    for i in range(len(inputs)):
        for j in range(i+1, len(inputs)):
            left, right = set(allmods[i]), set(allmods[j])
            print(f"Overlap {inputs[i].name} <-> {inputs[j].name}: {len(left & right)} shared; {len(left-right)} left-only; {len(right-left)} right-only")
            if not left & right: print("  WARNING: zero exact key overlap; inspect naming before merging.")
    stems = sorted(set().union(*(set(m) for m in allmods)))
    for stem in stems:
        present = [m[stem] for m in allmods if stem in m]
        a0, b0 = present[0]["A"], present[0]["B"]
        for m in present[1:]:
            if m["A"].shape[1] != a0.shape[1] or m["B"].shape[0] != b0.shape[0]:
                raise ValueError(f"Non-rank dimensions mismatch for {stem}; no output written.")
    print(f"\nUnion: {len(stems)} module stems; all detected shapes compatible.")
    activation_override = getattr(args, "activation_text", None)
    forge_metadata = (build_forge_user_metadata(inputs, activation_text=activation_override)
                      if write_json else None)
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]
    out, errs, ranks, bounds = {}, [], set(), []
    g_a, g_b, g_alpha, style_note = _pick_global_output_style(
        (e["from_lokr"], e) for m in allmods for e in m.values())
    if style_note:
        print(style_note)
    for n, stem in enumerate(stems, 1):
        check_cancel()
        entries = [m.get(stem) for m in allmods]
        As, Bs, e1s, ws = [], [], [], []
        for fi, (entry, weight) in enumerate(zip(entries, args.weights)):
            if entry is None: continue
            As.append(entry["A"].float())
            scale = entry["alpha_value"] / entry["rank"] * weight
            Bs.append(entry["B"].float() * scale)
            e1s.append(stage1_reports.get(fi, {}).get(stem, 0.0)); ws.append(weight)
        Ac, Bc = torch.cat(As, dim=0), torch.cat(Bs, dim=1)
        An, Bn, err, core_rank, tot_norm, tail_norm = repack_ex(Ac, Bc, args.target_rank, torch)
        bounds.append((combined_error_bound(e1s, ws, tot_norm, tail_norm), stem))
        if not torch.isfinite(An).all() or not torch.isfinite(Bn).all():
            raise FloatingPointError(f"Non-finite factor at {stem}")
        An, Bn = An.to(dtype).contiguous(), Bn.to(dtype).contiguous()
        if not torch.isfinite(An.float()).all() or not torch.isfinite(Bn.float()).all():
            raise FloatingPointError(f"{args.dtype} overflow at {stem}; try bf16 or fp32.")
        # One key style for the whole file (chosen once before this loop, input-order independent).
        a_suffix, b_suffix, alpha_suffix = g_a, g_b, g_alpha
        ak, bk = stem + a_suffix, stem + b_suffix
        if ak in out or bk in out: raise ValueError(f"Output key collision: {stem}")
        out[ak], out[bk] = An, Bn
        if alpha_suffix: out[stem + alpha_suffix] = torch.tensor(float(args.target_rank), dtype=torch.float32)
        ranks.add(int(An.shape[0])); errs.append((err, stem, core_rank))
        if n % 25 == 0 or n == len(stems): print(f"  processed {n}/{len(stems)}")
    meanerr = sum(x[0] for x in errs) / max(1, len(errs))
    maxerr = max((x[0] for x in errs), default=0.0)
    print(f"\nRelative Frobenius error from rank truncation only: mean={meanerr:.6g}; max={maxerr:.6g}")
    print("Largest module errors:")
    for err, stem, k in sorted(errs, reverse=True)[:10]: print(f"  {err:.6g} (core rank {k}) {stem}")
    print("This metric is relative to the (already LoKr-truncated) sum and excludes final dtype quantization error.")
    if any(any(v > 0 for v in r.values()) for r in stage1_reports.values()):
        known = [b for b, _ in bounds if b is not None]
        print(f"Combined error vs the EXACT sum (convert + merge truncation), triangle-inequality UPPER bound: "
              f"max={max(known):.4g}" if known else "Combined error bound not provable for any module.")
        if len(known) < len(bounds):
            print(f"  (no bound for {len(bounds) - len(known)} modules: stage-1 loss >= norm of the sum)")
        print("  Loose by construction; use --lokr-full-rank or --lokr-matrix-free to remove stage-1 loss.")
    metadata = {
        "format": "pt", **_output_identity_metadata(input_metadata),
        "ss_network_dim": str(args.target_rank),
        "ss_network_alpha": str(args.target_rank), "ss_output_name": output.stem,
        "modelspec.title": output.stem, "merge_method": "weighted_delta_sum_qr_small_core_svd",
        "merge_input_files": json.dumps([p.name for p in inputs], ensure_ascii=False),
        "merge_weights": json.dumps(args.weights), "merge_target_rank": str(args.target_rank),
        "merge_output_dtype": args.dtype,
        "merge_error_metric": "relative_frobenius_error_from_rank_truncation_only",
        **({"merge_lokr_alpha_semantics": LOKR_ALPHA_STATUS}
           if any(e["from_lokr"] for m_ in allmods for e in m_.values()) else {}),
    }
    check_cancel()  # last point where cancelling is possible; saving and validation are not interrupted
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        fd, name = tempfile.mkstemp(prefix=output.name+".", suffix=".tmp.safetensors", dir=str(output.parent))
        os.close(fd); temp = Path(name)
        save_file(out, str(temp), metadata=metadata)
        check = load_file(str(temp), device="cpu")
        if set(check) != set(out): raise RuntimeError("Serialized keys differ from in-memory keys.")
        for key, t in check.items():
            if tuple(t.shape) != tuple(out[key].shape): raise RuntimeError(f"Serialized shape mismatch: {key}")
            if not torch.isfinite(t.float()).all(): raise RuntimeError(f"Non-finite serialized tensor: {key}")
        if output.exists() and not args.overwrite: raise FileExistsError(f"Output appeared during processing: {output}")
        os.replace(str(temp), str(output)); temp = None
    finally:
        if temp is not None:
            try: temp.unlink(missing_ok=True)
            except OSError: pass
    print(f"\nSUCCESS: {output}")
    print(f"Output tensors: {len(out)}; factor ranks: {sorted(ranks)}")
    print("Saved file was reopened and validated before atomic replacement.")
    print("All original LoRAs remain untouched. Test in Forge Neo at low strength first.")
    if forge_metadata is None:
        print("Forge Neo JSON skipped (--no-forge-json).")
    else:
        try:
            write_forge_user_metadata(output, forge_metadata, overwrite=args.overwrite)
            if activation_override is not None:
                print("Activation text was taken from the activation text field / --activation-text.")
            else:
                print("Activation text and negative text were collected from the source Forge JSON files "
                      "(missing files are skipped).")
        except Exception as exc:
            # The LoRA itself is already saved, so this must not look like a failed merge.
            print(f"  WARNING: The LoRA was saved, but the Forge Neo JSON could not be written: {exc}")



def merge_loras(args=None):
    """Run the merger with end-to-end timing and lightweight RSS sampling."""
    if args is None:
        args = args_parse()
    cpu_threads = int(getattr(args, "cpu_threads", 6))
    monitor = _ProcessMemoryMonitor(interval=0.05)
    started = time.perf_counter()
    status = "failed"
    monitor.start()
    try:
        result = _merge_loras_impl(args)
        status = "completed"
        return result
    finally:
        elapsed = time.perf_counter() - started
        monitor.stop()
        _print_run_diagnostics(status, elapsed, cpu_threads, monitor)


class _GuiStream(io.TextIOBase):
    """File-like object that forwards everything written to it (print output) to a callback."""
    def __init__(self, emit):
        self._emit = emit

    def writable(self):
        return True

    def write(self, s):
        if s:
            self._emit(s)
        return len(s)

    def flush(self):
        pass


# Which LoKr option the GUI ticks by itself when a LoKr file is added. "matrix_free" keeps a single
# (approximate) truncation and never expands the Kronecker product; "full_rank" is exact but RAM-heavy.
AUTO_LOKR_OPTION = "matrix_free"


def lokr_detected(header_infos):
    """Names of inputs whose header says LoKr (is_lokr_state_keys-style detection, headers only)."""
    return [Path(info["path"]).name if info.get("path") else "?"
            for info in header_infos if info.get("format") == "lokr"]


class Krea2MergerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Krea2 SVD LoRA Merger")
        self.root.geometry("900x860")
        self.root.minsize(760, 700)

        self.input_files = []
        self.weight_vars = []
        self.header_infos = []
        self.output_file = tk.StringVar()
        self.target_rank = tk.StringVar(value="32")
        self.cpu_threads = tk.StringVar(value="6")
        self.lokr_full_rank = tk.BooleanVar(value=False)
        self.lokr_matrix_free = tk.BooleanVar(value=False)
        self.dtype = tk.StringVar(value="bf16")
        self.overwrite = tk.BooleanVar(value=False)
        self.dry_run = tk.BooleanVar(value=False)
        self.write_json = tk.BooleanVar(value=True)
        self.activation_dirty = False   # True once the user edits the activation text by hand
        self._autofilling = False
        self._lokr_user_override = False   # user clicked a LoKr checkbox: never touch them again this session
        self._lokr_auto_set = False        # the current tick was set by the program, not by the user
        self.running = False
        self.closing = False
        self.cancel_event = threading.Event()
        # The worker thread never touches Tk: it only puts items into this queue,
        # and the main thread drains it from _poll_queue().
        self.msg_queue = queue.Queue()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._poll_queue)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(6, weight=2)

        inp = ttk.Labelframe(outer, text="Input LoRA files", padding=8)
        inp.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        inp.columnconfigure(0, weight=1)
        inp.rowconfigure(0, weight=1)

        self.input_listbox = tk.Listbox(inp, height=7, exportselection=False)
        self.input_listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(inp, orient="vertical", command=self.input_listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.input_listbox.configure(yscrollcommand=scroll.set)

        controls = ttk.Frame(inp)
        controls.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        ttk.Button(controls, text="Add files…", command=self.add_input_files, **bs("primary")).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Remove selected", command=self.remove_input, **bs("danger-outline")).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Clear all", command=self.clear_inputs, **bs("secondary-outline")).pack(fill=tk.X, pady=2)
        ttk.Label(controls, text="Files are checked\nwhen added", justify="center").pack(fill=tk.X, pady=(10, 2))

        weights = ttk.Labelframe(outer, text="Merge coefficients", padding=8)
        weights.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        weights.columnconfigure(0, weight=1)
        self.weights_frame = ttk.Frame(weights)
        self.weights_frame.grid(row=0, column=0, sticky="ew")

        paths = ttk.Labelframe(outer, text="Files", padding=8)
        paths.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        paths.columnconfigure(1, weight=1)
        ttk.Label(paths, text="Output file:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(paths, textvariable=self.output_file).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(paths, text="Save as…", command=self.browse_output, **bs("secondary")).grid(row=0, column=2, pady=3)

        settings = ttk.Labelframe(outer, text="Settings", padding=8)
        settings.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        row = ttk.Frame(settings)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Target rank:").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=4096, textvariable=self.target_rank, width=8).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(row, text="Output dtype:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.dtype, values=("bf16", "fp16", "fp32"),
                     state="readonly", width=8).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="CPU threads:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(row, from_=1, to=20, textvariable=self.cpu_threads, width=5).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(settings, text="Preserve full LoKr rank until the final SVD (RAM intensive; preflight limit 2 GiB)",
                        variable=self.lokr_full_rank, command=self._on_lokr_option_clicked).pack(anchor="w")
        ttk.Checkbutton(settings, text="Matrix-free joint SVD for LoKr + LoRA (approximate; avoids Kronecker expansion)",
                        variable=self.lokr_matrix_free, command=self._on_lokr_option_clicked).pack(anchor="w")
        ttk.Checkbutton(settings, text="Allow overwriting existing output", variable=self.overwrite).pack(anchor="w")
        ttk.Checkbutton(settings, text="Dry run (validate only; do not save)", variable=self.dry_run).pack(anchor="w")
        ttk.Checkbutton(settings, text="Write Forge Neo sidecar JSON next to the output",
                        variable=self.write_json).pack(anchor="w")

        triggers = ttk.Labelframe(outer, text="Forge Neo activation text (editable)", padding=8)
        triggers.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        triggers.columnconfigure(0, weight=1)
        ttk.Label(
            triggers,
            text="Auto-collected from input LoRA JSON files. Edit freely; this text is saved to the output JSON."
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Button(triggers, text="Reload from JSON", command=lambda: self.refresh_activation_text(force=True),
                   **bs("secondary-outline")).grid(row=0, column=1, sticky="e", pady=(0, 4))
        trigger_row = ttk.Frame(triggers)
        trigger_row.grid(row=1, column=0, sticky="ew")
        trigger_row.columnconfigure(0, weight=1)
        self.activation_text = tk.Text(trigger_row, height=3, wrap="word", undo=True)
        self.activation_text.grid(row=0, column=0, sticky="ew")
        trigger_scroll = ttk.Scrollbar(trigger_row, orient="vertical", command=self.activation_text.yview)
        trigger_scroll.grid(row=0, column=1, sticky="ns")
        self.activation_text.configure(yscrollcommand=trigger_scroll.set)
        self.activation_text.bind("<<Modified>>", self._on_activation_modified)

        actions = ttk.Frame(outer)
        actions.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.run_button = ttk.Button(actions, text="Run merger", command=self.start_merge, **bs("success"))
        self.run_button.pack(side=tk.LEFT)
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel_merge,
                                        state="disabled", **bs("danger-outline"))
        self.cancel_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Close", command=self.on_close, **bs("secondary-outline")).pack(side=tk.LEFT, padx=8)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(actions, textvariable=self.status).pack(side=tk.LEFT, padx=12)
        if HAVE_BOOTSTRAP:
            self.theme_var = tk.StringVar(value=DEFAULT_THEME)
            theme_box = ttk.Combobox(actions, textvariable=self.theme_var, values=THEMES,
                                     state="readonly", width=10)
            theme_box.pack(side=tk.RIGHT)
            theme_box.bind("<<ComboboxSelected>>", self._on_theme)
            ttk.Label(actions, text="Theme:").pack(side=tk.RIGHT, padx=(0, 6))
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=120, **bs("success-striped"))
        self.progress.pack(side=tk.RIGHT, padx=12)

        logframe = ttk.Labelframe(outer, text="Run log", padding=6)
        logframe.grid(row=6, column=0, sticky="nsew")
        logframe.rowconfigure(0, weight=1)
        logframe.columnconfigure(0, weight=1)
        self.log = tk.Text(logframe, height=10, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        logscroll = ttk.Scrollbar(logframe, orient="vertical", command=self.log.yview)
        logscroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=logscroll.set)
        self._style_native_widgets()

    def _style_native_widgets(self):
        """tk.Listbox and tk.Text are not themed by ttk, so colour them from the active theme."""
        if not HAVE_BOOTSTRAP:
            return
        try:
            c = self.root.style.colors
            common = dict(bg=c.inputbg, fg=c.inputfg, relief="flat", borderwidth=0,
                          highlightthickness=1, highlightbackground=c.border, highlightcolor=c.primary)
            self.input_listbox.configure(selectbackground=c.primary, selectforeground=c.selectfg, **common)
            self.log.configure(insertbackground=c.inputfg, font=("Consolas", 9), **common)
            self.activation_text.configure(insertbackground=c.inputfg, **common)
        except Exception:
            pass  # purely cosmetic; never block the GUI

    def _on_theme(self, _event=None):
        self.root.style.theme_use(self.theme_var.get())
        self._style_native_widgets()

    # ---- logging -------------------------------------------------------
    def append_raw(self, text):
        """Append text exactly as given (used for streamed print output)."""
        if self.closing:
            return
        self.log.configure(state="normal")
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def append_log(self, text):
        self.append_raw(text if text.endswith("\n") else text + "\n")

    def _poll_queue(self):
        """Main-thread pump: apply log text and completion events from the worker."""
        try:
            while True:
                item = self.msg_queue.get_nowait()
                if item[0] == "log":
                    self.append_raw(item[1])
                elif item[0] == "done":
                    self.finish_run(item[1], item[2])
        except queue.Empty:
            pass
        except tk.TclError:
            return  # window destroyed
        try:
            self.root.after(50, self._poll_queue)
        except (tk.TclError, RuntimeError):
            pass

    # ---- input list ----------------------------------------------------
    def add_input_files(self):
        paths = filedialog.askopenfilenames(
            title="Select LoRA files",
            filetypes=(("Safetensors files", "*.safetensors"), ("All files", "*.*"))
        )
        added, rejected = [], []
        for raw_path in paths:
            if raw_path in self.input_files:
                continue
            try:
                info = inspect_lora_header(raw_path)
                cross_warnings = compare_lora_headers(info, self.header_infos)
                warnings = info["warnings"] + cross_warnings
                if warnings:
                    details = (
                        f"File: {Path(raw_path).name}\n"
                        f"Tensors: {info['tensor_count']}\n"
                        f"LoRA layers: {info['module_count']}\n"
                        f"Ranks: {info['ranks']}\n"
                        f"Data types: {', '.join(info['dtypes']) or 'unknown'}\n\n"
                        "Warnings:\n• " + "\n• ".join(warnings) +
                        "\n\nAdd file?"
                    )
                    if not messagebox.askyesno("LoRA check with warnings", details, parent=self.root):
                        rejected.append(Path(raw_path).name + " (user refusal)")
                        continue
                self.input_files.append(raw_path)
                self.weight_vars.append(tk.StringVar(value="1.0"))
                self.header_infos.append(info)
                added.append(Path(raw_path).name)
                for note in info["notes"]:
                    self.append_log(f"{Path(raw_path).name}: {note}")
            except Exception as exc:
                rejected.append(Path(raw_path).name + f" — {exc}")
                messagebox.showerror(
                    "Invalid LoRA",
                    f"File was not added to the list:\n{Path(raw_path).name}\n\n{exc}\n\n"
                    "Only the SafeTensors structure was checked. The file was not modified.",
                    parent=self.root,
                )
        self.refresh_inputs()
        if added:
            self.refresh_activation_text()
            self._sync_lokr_options()
            self.status.set(f"Added after check: {len(added)}")
        elif not paths:
            return
        elif rejected:
            self.status.set(f"Not added: {len(rejected)}")

    def _on_lokr_option_clicked(self):
        self._lokr_user_override = True
        self._lokr_auto_set = False

    def _ask_lokr_confirm(self, names, label):
        """Ask once whether to tick the LoKr option. Separate method so it can be replaced in tests."""
        return messagebox.askyesno(
            "LoKr detected",
            f"LoKr file detected: {', '.join(names)}\n\n"
            f"Tick \"{label}\"?\n\n"
            "Yes: LoKr is truncated only once (approximate, no Kronecker expansion, lower RAM use).\n"
            "No: the default two-step conversion is used (LoKr -> target rank -> final rank), "
            "which adds a second truncation whose loss the merge report does not include.\n\n"
            "You can change this later with the checkboxes. For the exact but RAM-heavy variant, "
            "tick \"Preserve full LoKr rank\" yourself.",
            parent=self.root,
        )

    def _sync_lokr_options(self):
        """Offer to tick the LoKr option when a LoKr input is present (only after the user agrees).

        Never overrides a manual choice: if the user clicked either LoKr checkbox, declined the offer,
        or one of the options is already ticked, nothing is changed and nothing is asked. A tick set after
        consent is cleared again when the last LoKr input is removed."""
        names = lokr_detected(self.header_infos)
        var = self.lokr_matrix_free if AUTO_LOKR_OPTION == "matrix_free" else self.lokr_full_rank
        label = ("Matrix-free joint SVD" if AUTO_LOKR_OPTION == "matrix_free"
                 else "Preserve full LoKr rank")
        if names:
            if (self._lokr_user_override or self._lokr_auto_set
                    or self.lokr_full_rank.get() or self.lokr_matrix_free.get()):
                return
            if self._ask_lokr_confirm(names, label):
                var.set(True)
                self._lokr_auto_set = True
                self.append_log(f"LoKr detected ({', '.join(names)}): '{label}' ticked with your confirmation "
                                "(LoKr is truncated only once).")
            else:
                self._lokr_user_override = True          # declined: do not ask again this session
                self.append_log(f"LoKr detected ({', '.join(names)}): default two-step conversion kept "
                                "(adds a second truncation).")
        elif self._lokr_auto_set:
            var.set(False)
            self._lokr_auto_set = False
            self.append_log("No LoKr inputs left: the LoKr option ticked earlier was cleared.")

    def refresh_inputs(self):
        self.input_listbox.delete(0, tk.END)
        for path in self.input_files:
            self.input_listbox.insert(tk.END, Path(path).name)
        for child in self.weights_frame.winfo_children():
            child.destroy()
        if not self.input_files:
            ttk.Label(self.weights_frame, text="Add one or more LoRA files above.").grid(row=0, column=0, sticky="w")
            return
        for i, path in enumerate(self.input_files):
            row = ttk.Frame(self.weights_frame)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.columnconfigure(0, weight=1)
            ttk.Label(row, text=Path(path).name).grid(row=0, column=0, sticky="w")
            ttk.Label(row, text="Coefficient:").grid(row=0, column=1, padx=(10, 4))
            ttk.Spinbox(
                row,
                from_=-100.0,
                to=100.0,
                increment=0.1,
                textvariable=self.weight_vars[i],
                width=10,
                format="%.2f",
            ).grid(row=0, column=2)

    def _on_activation_modified(self, _event=None):
        """Remember that the user edited the field, so automatic refills do not overwrite the edit."""
        if not self.activation_text.edit_modified():
            return
        self.activation_text.edit_modified(False)
        if not self._autofilling:
            self.activation_dirty = True

    def refresh_activation_text(self, force=False):
        """Rebuild the editable trigger field from the selected sidecar JSON files.

        Hand-edited text is kept unless force=True (the "Reload from JSON" button).
        """
        if self.activation_dirty and not force:
            self.append_log("Activation text was edited manually and was not replaced "
                            "(use \"Reload from JSON\" to rebuild it).")
            return
        combined = _merge_text_fields([
            _read_forge_user_metadata(path).get("activation text", "")
            for path in self.input_files
        ])
        self._autofilling = True
        try:
            self.activation_text.delete("1.0", tk.END)
            self.activation_text.insert("1.0", combined)
            self.activation_text.edit_modified(False)
            self.activation_text.edit_reset()
        finally:
            self._autofilling = False
        self.activation_dirty = False
        if not combined and self.input_files:
            self.append_log("Forge metadata: no activation text found in selected input JSON files.")

    def remove_input(self):
        selected = self.input_listbox.curselection()
        if not selected:
            return
        i = selected[0]
        del self.input_files[i]
        del self.weight_vars[i]
        del self.header_infos[i]
        self.refresh_inputs()
        self.refresh_activation_text()
        self._sync_lokr_options()

    def clear_inputs(self):
        self.input_files.clear()
        self.weight_vars.clear()
        self.header_infos.clear()
        self.refresh_inputs()
        self.refresh_activation_text()
        self._sync_lokr_options()

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Choose output LoRA file",
            defaultextension=".safetensors",
            filetypes=(("Safetensors files", "*.safetensors"), ("All files", "*.*"))
        )
        if path:
            self.output_file.set(path)

    # ---- running -------------------------------------------------------
    def validate_and_build_args(self):
        if not self.input_files:
            raise ValueError("Add at least one input LoRA file.")
        for path in self.input_files:
            if not Path(path).is_file():
                raise ValueError(f"Input file does not exist:\n{path}")

        output_text = self.output_file.get().strip()
        if not output_text:
            raise ValueError("Choose an output file.")
        output = Path(output_text).expanduser()
        if output.suffix.lower() != ".safetensors":
            raise ValueError("Output file must have the .safetensors extension.")
        if output.resolve() in {Path(p).resolve() for p in self.input_files}:
            raise ValueError("Output file must not be one of the input files.")
        if output.exists() and not self.overwrite.get() and not self.dry_run.get():
            raise ValueError("Output file already exists. Choose another name or enable overwrite.")
        output_json = output.with_suffix(".json")
        if (self.write_json.get() and not self.dry_run.get() and not self.overwrite.get()
                and output_json.exists()):
            raise ValueError(f"Forge metadata JSON already exists:\n{output_json}\n\n"
                             "Choose another name, enable overwrite, or untick the Forge JSON option.")

        if self.lokr_full_rank.get() and self.lokr_matrix_free.get():
            raise ValueError("\"Preserve full LoKr rank\" and \"Matrix-free joint SVD\" "
                             "are mutually exclusive; untick one of them.")
        try:
            rank = int(self.target_rank.get())
        except (ValueError, tk.TclError):
            raise ValueError("Target rank must be a whole number.")
        if rank < 1:
            raise ValueError("Target rank must be at least 1.")
        try:
            cpu_threads = int(self.cpu_threads.get())
        except (ValueError, tk.TclError):
            raise ValueError("CPU threads must be a whole number from 1 to 20.")
        if not 1 <= cpu_threads <= 20:
            raise ValueError("CPU threads must be between 1 and 20.")
        try:
            weights = [float(v.get()) for v in self.weight_vars]
        except (ValueError, tk.TclError):
            raise ValueError("Every coefficient must be a valid number, for example 1.0 or 0.7.")
        if not all(math.isfinite(w) for w in weights):
            raise ValueError("Coefficients must be finite numbers.")
        if len(weights) != len(self.input_files):
            raise ValueError("Supply exactly one coefficient per input LoRA.")

        text = self.activation_text.get("1.0", "end-1c").strip()
        # None = let the merger collect it from the source JSON files; "" only if the user cleared it on purpose.
        activation_text = text if (text or self.activation_dirty) else None

        return argparse.Namespace(
            inputs=list(self.input_files),
            output=str(output),
            weights=weights,
            target_rank=rank,
            cpu_threads=cpu_threads,
            lokr_full_rank=self.lokr_full_rank.get(),
            lokr_matrix_free=self.lokr_matrix_free.get(),
            lokr_svd_oversample=8,
            lokr_svd_power_iterations=2,
            lokr_workspace_max_gb=2.0,
            lokr_max_factor_gb=2.0,
            dtype=self.dtype.get(),
            dry_run=self.dry_run.get(),
            overwrite=self.overwrite.get(),
            no_pause=True,
            cancel_event=self.cancel_event,
            activation_text=activation_text,
            no_forge_json=not self.write_json.get(),
        )

    def start_merge(self):
        if self.running:
            return
        try:
            args = self.validate_and_build_args()
        except Exception as exc:
            messagebox.showerror("Check settings", str(exc), parent=self.root)
            return
        self.running = True
        self.cancel_event.clear()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status.set("Running…")
        self.progress.start(12)
        self.append_log("\n" + "=" * 70)
        self.append_log("Running merger directly inside this application (no subprocess).")
        self.append_log(f"Inputs: {len(args.inputs)}; rank={args.target_rank}; dtype={args.dtype}; "
                        f"CPU threads={args.cpu_threads}; dry_run={args.dry_run}")
        threading.Thread(target=self.run_worker, args=(args,), daemon=True).start()

    def run_worker(self, args):
        # Output is streamed to the log window line by line while the merge runs.
        stream = _GuiStream(lambda s: self.msg_queue.put(("log", s)))
        try:
            with contextlib.redirect_stdout(stream):
                merge_loras(args)
            self.msg_queue.put(("done", 0, ""))
        except InterruptedError:
            self.msg_queue.put(("done", 2, "\nMerge cancelled. No output file was written."))
        except Exception:
            self.msg_queue.put(("done", 1, "\n--- ERROR ---\n" + traceback.format_exc()))

    def finish_run(self, returncode, extra):
        self.running = False
        if self.closing:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            return
        self.progress.stop()
        self.cancel_button.configure(state="disabled")
        if extra.strip():
            self.append_log(extra.rstrip())
        self.run_button.configure(state="normal")
        if returncode == 2:
            self.status.set("Cancelled.")
        elif returncode == 0:
            self.status.set("Finished successfully.")
            messagebox.showinfo(
                "Merger finished",
                "The script finished successfully.\nCheck the run log for the output path and validation details.",
                parent=self.root
            )
        else:
            self.status.set(f"Failed (exit code {returncode}).")
            messagebox.showerror(
                "Merger failed",
                "The merger returned an error. The full output is shown in the Run log below.",
                parent=self.root
            )

    def cancel_merge(self):
        if not self.running or self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status.set("Cancellation requested; waiting for the current step to finish…")

    def on_close(self):
        if self.running:
            if not messagebox.askokcancel(
                "Merger is running",
                "Request cancellation and close after the current layer finishes? "
                "No output file will be written if cancellation completes before saving.",
                parent=self.root,
            ):
                return
            self.closing = True
            self.cancel_event.set()
            self.status.set("Cancellation requested; waiting for the current layer to finish…")
            self.run_button.configure(state="disabled")
            return
        self.closing = True
        self.root.destroy()


def launch_gui():
    if not HAVE_TK:
        raise RuntimeError("tkinter is not available in this Python installation. "
                           "Run the script with command-line arguments instead (see --help).")
    if HAVE_BOOTSTRAP:
        root = ttk.Window(themename=DEFAULT_THEME)
    else:
        print("Tip: pip install ttkbootstrap for a modern look (using plain ttk for now).", file=sys.stderr)
        root = tk.Tk()
    Krea2MergerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        try:
            launch_gui()
        except Exception as exc:
            print(f"Could not start GUI: {exc}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
    else:
        # --help and argument errors exit here, before any "press Enter" prompt.
        cli_args = args_parse()
        ok = True
        try:
            merge_loras(cli_args)
            print("\nFinished.")
        except Exception as exc:
            ok = False
            print(f"\nERROR: {exc}", file=sys.stderr)
            traceback.print_exc()
        if not ok and not cli_args.no_pause and sys.stdin and sys.stdin.isatty():
            try:
                input("\nPress Enter to close...")
            except EOFError:
                pass
        sys.exit(0 if ok else 1)
