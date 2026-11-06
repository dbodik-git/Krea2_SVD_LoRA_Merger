#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Krea2 SVD LoRA Merger v4: CPU merge, QR + small-core SVD, atomic save."""
import argparse, json, math, os, sys, tempfile, traceback, contextlib, io, struct
import threading
from collections import defaultdict
from pathlib import Path

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

D = Path(r"E:\SD\Lora\krea2")
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
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-pause", action="store_true",
                   help="do not wait for Enter after an error (for scripts and .bat files)")
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
        if aa and ab and aa != ab:
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


def load_lora(path, torch, load_file):
    raw = load_file(str(path), device="cpu")
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



FORGE_USER_METADATA_KEYS = (
    "description", "sd version", "activation text",
    "preferred weight", "negative text", "notes",
)


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


def build_forge_user_metadata(input_paths, output_path):
    """Build Forge Neo's six-key user metadata from input LoRA sidecar JSON files."""
    sources = [_read_forge_user_metadata(path) for path in input_paths]
    names = [Path(path).stem for path in input_paths]
    notes = [item.get("notes", "").strip() for item in sources
             if isinstance(item.get("notes"), str) and item.get("notes").strip()]
    weights = []
    for item in sources:
        try:
            weight = float(item.get("preferred weight", 0.0))
            if math.isfinite(weight):
                weights.append(weight)
        except (TypeError, ValueError):
            pass
    preferred_weight = sum(weights) / len(weights) if weights else 0.0
    metadata = {
        "description": "Merged LoRA: " + ", ".join(names),
        "sd version": "krea",
        "activation text": _merge_text_fields([item.get("activation text", "") for item in sources]),
        "preferred weight": preferred_weight,
        "negative text": _merge_text_fields([item.get("negative text", "") for item in sources]),
        "notes": "Merged from: " + ", ".join(names)
                 + (("\nSource notes:\n" + "\n".join(notes)) if notes else ""),
    }
    return {key: metadata[key] for key in FORGE_USER_METADATA_KEYS}


def write_forge_user_metadata(output_path, metadata, overwrite=False):
    """Atomically write Forge Neo-compatible sidecar JSON beside the merged LoRA."""
    json_path = Path(output_path).with_suffix(".json")
    if json_path.exists() and not overwrite:
        raise FileExistsError(
            f"Forge metadata JSON already exists: {json_path}. "
            "Enable overwrite or choose another output name."
        )
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

def merge_loras(args=None):
    if args is None:
        args = args_parse()
    if not args.inputs: raise ValueError("At least one input is required.")
    if getattr(args, "weights", None) is None:
        args.weights = [1.0] * len(args.inputs)
    if len(args.weights) != len(args.inputs): raise ValueError("Supply exactly one --weights value per input.")
    if args.target_rank < 1: raise ValueError("--target-rank must be >= 1.")
    if not all(math.isfinite(w) for w in args.weights): raise ValueError("Weights must be finite.")
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    inputs = [Path(x).expanduser() for x in args.inputs]
    output = Path(args.output).expanduser()
    if output.resolve() in {p.resolve() for p in inputs}: raise ValueError("Output cannot overwrite an input.")
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing: raise FileNotFoundError("Missing input(s):\n  " + "\n  ".join(missing))
    # A dry run never writes anything, so an existing output file is not a problem there.
    if output.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output exists: {output}. Choose another --output or pass --overwrite.")
    output_json = output.with_suffix(".json")
    if output_json.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Forge metadata JSON exists: {output_json}. Choose another output or pass --overwrite.")
    print("Krea2 SVD LoRA Merger")
    print(f"Target rank={args.target_rank}; dtype={args.dtype}; sources remain read-only.")
    for p, w in zip(inputs, args.weights): print(f"  {p.name}  coefficient={w:g}")
    print(f"Output: {output}\n")
    allmods, metadatas = [], []
    for p in inputs:
        with safe_open(str(p), framework="pt", device="cpu") as sf: md = sf.metadata() or {}
        mods = load_lora(p, torch, load_file)
        allmods.append(mods); metadatas.append(md)
        print(f"Read {p.name}: {len(mods)} modules; architecture={md.get('modelspec.architecture','unspecified')}; base={md.get('ss_base_model_version','unspecified')}")
        # Without alpha tensors the scale is assumed to be 1 (alpha = rank). Warn if metadata disagrees.
        no_alpha_ranks = {m["rank"] for m in mods.values() if "alpha" not in m}
        meta_alpha = md.get("ss_network_alpha")
        if no_alpha_ranks and meta_alpha is not None:
            try:
                meta_alpha = float(meta_alpha)
            except (TypeError, ValueError):
                meta_alpha = None
            if meta_alpha is not None and any(abs(meta_alpha - r) > 1e-9 for r in no_alpha_ranks):
                print(f"  WARNING: {p.name} has no alpha tensors, so scale 1 (alpha = rank) is assumed, "
                      f"but metadata says ss_network_alpha={meta_alpha:g}. The contribution may be scaled incorrectly.")
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
    if args.dry_run:
        print("DRY RUN complete. No output written."); return
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]
    out, errs, ranks = {}, [], set()
    cancel_event = getattr(args, "cancel_event", None)
    for n, stem in enumerate(stems, 1):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Merge cancelled by user. No output file was written.")
        entries = [m.get(stem) for m in allmods]
        As, Bs = [], []
        for entry, weight in zip(entries, args.weights):
            if entry is None: continue
            As.append(entry["A"].float())
            scale = entry["alpha_value"] / entry["rank"] * weight
            Bs.append(entry["B"].float() * scale)
        Ac, Bc = torch.cat(As, dim=0), torch.cat(Bs, dim=1)
        An, Bn, err, core_rank = repack(Ac, Bc, args.target_rank, torch)
        if not torch.isfinite(An).all() or not torch.isfinite(Bn).all():
            raise FloatingPointError(f"Non-finite factor at {stem}")
        An, Bn = An.to(dtype).contiguous(), Bn.to(dtype).contiguous()
        if not torch.isfinite(An.float()).all() or not torch.isfinite(Bn.float()).all():
            raise FloatingPointError(f"{args.dtype} overflow at {stem}; try bf16 or fp32.")
        # Preserve first encountered A/B suffix style and alpha-key style for this module.
        first = next(e for e in entries if e is not None)
        ak, bk = stem + first["A_suffix"], stem + first["B_suffix"]
        if ak in out or bk in out: raise ValueError(f"Output key collision: {stem}")
        out[ak], out[bk] = An, Bn
        alpha_suffix = next((e["alpha_suffix"] for e in entries if e is not None and "alpha" in e), None)
        if alpha_suffix: out[stem + alpha_suffix] = torch.tensor(float(args.target_rank), dtype=torch.float32)
        ranks.add(int(An.shape[0])); errs.append((err, stem, core_rank))
        if n % 25 == 0 or n == len(stems): print(f"  processed {n}/{len(stems)}")
    meanerr = sum(x[0] for x in errs) / max(1, len(errs))
    maxerr = max((x[0] for x in errs), default=0.0)
    print(f"\nRelative Frobenius error from rank truncation only: mean={meanerr:.6g}; max={maxerr:.6g}")
    print("Largest module errors:")
    for err, stem, k in sorted(errs, reverse=True)[:10]: print(f"  {err:.6g} (core rank {k}) {stem}")
    print("This metric excludes final dtype quantization error.")
    metadata = {
        "format": "pt", "modelspec.architecture": "krea2/lora",
        "ss_base_model_version": "krea2", "ss_network_dim": str(args.target_rank),
        "ss_network_alpha": str(args.target_rank), "ss_output_name": output.stem,
        "modelspec.title": output.stem, "merge_method": "weighted_delta_sum_qr_small_core_svd",
        "merge_input_files": json.dumps([p.name for p in inputs], ensure_ascii=False),
        "merge_weights": json.dumps(args.weights), "merge_target_rank": str(args.target_rank),
        "merge_output_dtype": args.dtype,
        "merge_error_metric": "relative_frobenius_error_from_rank_truncation_only",
    }
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
    forge_metadata = build_forge_user_metadata(inputs, output)
    write_forge_user_metadata(output, forge_metadata, overwrite=args.overwrite)
    print("Activation text and negative text were collected from source Forge JSON files (missing files are skipped).")


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


class Krea2MergerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Krea2 SVD LoRA Merger")
        self.root.geometry("900x760")
        self.root.minsize(760, 620)

        self.input_files = []
        self.weight_vars = []
        self.header_infos = []
        self.output_file = tk.StringVar()
        self.target_rank = tk.StringVar(value="32")
        self.dtype = tk.StringVar(value="bf16")
        self.overwrite = tk.BooleanVar(value=False)
        self.dry_run = tk.BooleanVar(value=False)
        self.running = False
        self.closing = False
        self.cancel_event = threading.Event()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(5, weight=2)

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
        ttk.Checkbutton(settings, text="Allow overwriting existing output", variable=self.overwrite).pack(anchor="w")
        ttk.Checkbutton(settings, text="Dry run (validate only; do not save)", variable=self.dry_run).pack(anchor="w")

        actions = ttk.Frame(outer)
        actions.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.run_button = ttk.Button(actions, text="Run merger", command=self.start_merge, **bs("success"))
        self.run_button.pack(side=tk.LEFT)
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
        logframe.grid(row=5, column=0, sticky="nsew")
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

    def _post(self, func, *a):
        """Schedule callbacks on the Tk main thread; ignore errors after destruction."""
        try:
            self.root.after(0, func, *a)
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
            self.status.set(f"Added after check: {len(added)}")
        elif not paths:
            return
        elif rejected:
            self.status.set(f"Not added: {len(rejected)}")

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

    def remove_input(self):
        selected = self.input_listbox.curselection()
        if not selected:
            return
        i = selected[0]
        del self.input_files[i]
        del self.weight_vars[i]
        del self.header_infos[i]
        self.refresh_inputs()

    def clear_inputs(self):
        self.input_files.clear()
        self.weight_vars.clear()
        self.header_infos.clear()
        self.refresh_inputs()

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
        if output.resolve() in {Path(p).resolve() for p in self.input_files}:
            raise ValueError("Output file must not be one of the input files.")
        if output.exists() and not self.overwrite.get() and not self.dry_run.get():
            raise ValueError("Output file already exists. Choose another name or enable overwrite.")

        try:
            rank = int(self.target_rank.get())
        except (ValueError, tk.TclError):
            raise ValueError("Target rank must be a whole number.")
        if rank < 1:
            raise ValueError("Target rank must be at least 1.")
        try:
            weights = [float(v.get()) for v in self.weight_vars]
        except (ValueError, tk.TclError):
            raise ValueError("Every coefficient must be a valid number, for example 1.0 or 0.7.")
        if not all(math.isfinite(w) for w in weights):
            raise ValueError("Coefficients must be finite numbers.")
        if len(weights) != len(self.input_files):
            raise ValueError("Supply exactly one coefficient per input LoRA.")

        return argparse.Namespace(
            inputs=list(self.input_files),
            output=str(output),
            weights=weights,
            target_rank=rank,
            dtype=self.dtype.get(),
            dry_run=self.dry_run.get(),
            overwrite=self.overwrite.get(),
            no_pause=True,
            cancel_event=self.cancel_event,
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
        self.run_button.configure(state="disabled")
        self.status.set("Running…")
        self.progress.start(12)
        self.append_log("\n" + "=" * 70)
        self.append_log("Running merger directly inside this application (no subprocess).")
        self.append_log(f"Inputs: {len(args.inputs)}; rank={args.target_rank}; dtype={args.dtype}; dry_run={args.dry_run}")
        threading.Thread(target=self.run_worker, args=(args,), daemon=True).start()

    def run_worker(self, args):
        # Output is streamed to the log window line by line while the merge runs.
        stream = _GuiStream(lambda s: self._post(self.append_raw, s))
        try:
            with contextlib.redirect_stdout(stream):
                merge_loras(args)
            self._post(self.finish_run, 0, "")
        except Exception:
            self._post(self.finish_run, 1, "\n--- ERROR ---\n" + traceback.format_exc())

    def finish_run(self, returncode, extra):
        self.running = False
        if self.closing:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            return
        self.progress.stop()
        if extra.strip():
            self.append_log(extra.rstrip())
        self.run_button.configure(state="normal")
        if returncode == 0:
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
