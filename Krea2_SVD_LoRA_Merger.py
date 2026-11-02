#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Krea2 SVD LoRA Merger v3: CPU merge, QR + small-core SVD, atomic save."""
import argparse, json, math, os, sys, tempfile, traceback, contextlib, io, struct
from collections import defaultdict
from pathlib import Path

D = Path(r"E:\SD\Lora\krea2")
DEFAULT_INPUTS = [
    D / "Krea2_TextFusion_Refusal_Reduction_v2_SVD_r17.safetensors",
    D / "AltGirlKreaV1.5_k_AltG1rl_SVD_r14.safetensors",
    D / "RealisticGoodGirlKrea2_krea2_k_rggl_SVD_r14.safetensors",
]
DEFAULT_OUTPUT = D / "Krea2_Merged_3x_SVD_r32_v3.safetensors"
A_SUFFIXES = (".lora_A.default.weight", ".lora_down.weight", ".lora_A.weight", ".lora_A.default", ".lora_A")
B_SUFFIXES = (".lora_B.default.weight", ".lora_up.weight", ".lora_B.weight", ".lora_B.default", ".lora_B")
ALPHA_SUFFIXES = (".lora_alpha", ".alpha")

def args_parse():
    p = argparse.ArgumentParser(description="Merge Krea2 LoRAs and SVD-repack to a uniform rank.")
    p.add_argument("--inputs", nargs="+", default=[str(x) for x in DEFAULT_INPUTS])
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--weights", nargs="+", type=float, default=[1.0, 1.0, 1.0])
    p.add_argument("--target-rank", type=int, default=32)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

MAX_HEADER_BYTES = 256 * 1024 * 1024

def inspect_lora_header(path):
    """Fast, CPU-only SafeTensors structural check; does not load tensor payloads."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Поддерживаются только файлы .safetensors.")
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("Файл слишком короткий: нет заголовка SafeTensors.")
        n = struct.unpack("<Q", prefix)[0]
        if n < 2 or n > MAX_HEADER_BYTES or 8 + n > size:
            raise ValueError(f"Недопустимая длина заголовка SafeTensors: {n} байт.")
        raw = f.read(n)
    try:
        header = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать JSON-заголовок: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("Неверная структура заголовка SafeTensors.")

    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("Поле __metadata__ имеет неверный формат.")

    tensors = {}
    for key, item in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(key, str) or not isinstance(item, dict):
            raise ValueError("В заголовке обнаружена некорректная запись тензора.")
        shape, dtype = item.get("shape"), item.get("dtype")
        offsets = item.get("data_offsets")
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            raise ValueError(f"Некорректная shape у тензора: {key}")
        if not isinstance(dtype, str) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Неполное описание тензора в заголовке: {key}")
        if not all(isinstance(x, int) for x in offsets) or offsets[0] < 0 or offsets[1] < offsets[0] or 8 + n + offsets[1] > size:
            raise ValueError(f"Некорректные data_offsets у тензора: {key}")
        tensors[key] = {"shape": tuple(shape), "dtype": dtype}

    mods, unknown = defaultdict(dict), []
    for key, info in tensors.items():
        parsed = parse_key(key)
        if parsed is None:
            unknown.append(key)
            continue
        kind, stem, suffix = parsed
        if kind in mods[stem]:
            raise ValueError(f"Дублируется компонент {kind} для слоя {stem}.")
        mods[stem][kind] = info
        mods[stem][kind + "_suffix"] = suffix

    errors = []
    if unknown:
        errors.append("Нераспознанные тензоры (программа не будет их отбрасывать): " + ", ".join(unknown[:8]))
    if not mods:
        errors.append("Не найдено ни одной поддерживаемой пары LoRA A/B.")
    ranks, dtypes = set(), set()
    for stem, m in mods.items():
        if "A" not in m or "B" not in m:
            errors.append(f"Неполная пара A/B: {stem}")
            continue
        ash, bsh = m["A"]["shape"], m["B"]["shape"]
        if len(ash) != 2 or len(bsh) != 2 or ash[0] != bsh[1] or ash[0] < 1:
            errors.append(f"Неверные формы A/B у {stem}: A={ash}, B={bsh}")
            continue
        ranks.add(ash[0]); dtypes.update((m["A"]["dtype"], m["B"]["dtype"]))
        if "alpha" in m and len(m["alpha"]["shape"]) > 1:
            errors.append(f"Alpha не скалярный у слоя {stem}: {m['alpha']['shape']}")

    if errors:
        raise ValueError("\n".join(errors))
    warnings = []
    if not metadata.get("modelspec.architecture"):
        warnings.append("В метаданных нет modelspec.architecture.")
    if not (metadata.get("ss_base_model_version") or metadata.get("ss_base_model")):
        warnings.append("В метаданных не указана базовая модель.")
    if len(ranks) > 1:
        warnings.append("Внутри LoRA используются разные ранги: " + ", ".join(map(str, sorted(ranks))))
    return {"path": path, "metadata": metadata, "modules": mods,
            "tensor_count": len(tensors), "module_count": len(mods),
            "ranks": sorted(ranks), "dtypes": sorted(dtypes), "warnings": warnings}


def compare_lora_headers(new_info, existing_infos):
    """Check shared target dimensions against already selected LoRAs."""
    warnings = []
    for old in existing_infos:
        shared = set(new_info["modules"]) & set(old["modules"])
        if not shared:
            warnings.append(f"Нет общих целевых слоёв с {old['path'].name}.")
            continue
        bad = []
        for stem in shared:
            a, b = new_info["modules"][stem], old["modules"][stem]
            if a["A"]["shape"][1] != b["A"]["shape"][1] or a["B"]["shape"][0] != b["B"]["shape"][0]:
                bad.append(stem)
        if bad:
            raise ValueError(
                f"Несовместимые размеры с {old['path'].name}: {len(bad)} общих слоёв.\n" +
                "Первые несовпадения: " + ", ".join(bad[:6])
            )
        ma, mb = new_info["metadata"], old["metadata"]
        aa, ab = ma.get("modelspec.architecture"), mb.get("modelspec.architecture")
        ba = ma.get("ss_base_model_version") or ma.get("ss_base_model")
        bb = mb.get("ss_base_model_version") or mb.get("ss_base_model")
        if aa and ab and aa != ab:
            warnings.append(f"Архитектуры в метаданных различаются с {old['path'].name}: {aa} / {ab}.")
        if ba and bb and ba != bb:
            warnings.append(f"Базовые модели в метаданных различаются с {old['path'].name}: {ba} / {bb}.")
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

def merge_loras(args=None):
    if args is None:
        args = args_parse()
    if not args.inputs: raise ValueError("At least one input is required.")
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
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Choose another --output or pass --overwrite.")
    print("Krea2 SVD LoRA Merger v3 — CPU preflight + uniform-rank SVD")
    print(f"Target rank={args.target_rank}; dtype={args.dtype}; sources remain read-only.")
    for p, w in zip(inputs, args.weights): print(f"  {p.name}  coefficient={w:g}")
    print(f"Output: {output}\n")
    allmods, metadatas = [], []
    for p in inputs:
        mods = load_lora(p, torch, load_file)
        with safe_open(str(p), framework="pt", device="cpu") as sf: md = sf.metadata() or {}
        allmods.append(mods); metadatas.append(md)
        print(f"Read {p.name}: {len(mods)} modules; architecture={md.get('modelspec.architecture','unspecified')}; base={md.get('ss_base_model_version','unspecified')}")
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
    for n, stem in enumerate(stems, 1):
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

# GUI dependencies are used only when launched without command-line arguments.
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class Krea2MergerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Krea2 SVD LoRA Merger v4 — preflight")
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

        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(5, weight=2)

        inp = ttk.LabelFrame(outer, text="Input LoRA files", padding=8)
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
        ttk.Button(controls, text="Add files…", command=self.add_input_files).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Remove selected", command=self.remove_input).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Clear all", command=self.clear_inputs).pack(fill=tk.X, pady=2)
        ttk.Label(controls, text="Проверка структуры\nперед добавлением", justify="center").pack(fill=tk.X, pady=(10, 2))

        weights = ttk.LabelFrame(outer, text="Merge coefficients", padding=8)
        weights.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        weights.columnconfigure(0, weight=1)
        self.weights_frame = ttk.Frame(weights)
        self.weights_frame.grid(row=0, column=0, sticky="ew")

        paths = ttk.LabelFrame(outer, text="Files", padding=8)
        paths.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        paths.columnconfigure(1, weight=1)
        ttk.Label(paths, text="Output file:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(paths, textvariable=self.output_file).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(paths, text="Save as…", command=self.browse_output).grid(row=0, column=2, pady=3)

        settings = ttk.LabelFrame(outer, text="Settings", padding=8)
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
        self.run_button = ttk.Button(actions, text="Run merger", command=self.start_merge)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(actions, text="Close", command=self.root.destroy).pack(side=tk.LEFT, padx=8)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(actions, textvariable=self.status).pack(side=tk.LEFT, padx=12)

        logframe = ttk.LabelFrame(outer, text="Run log", padding=6)
        logframe.grid(row=5, column=0, sticky="nsew")
        logframe.rowconfigure(0, weight=1)
        logframe.columnconfigure(0, weight=1)
        self.log = tk.Text(logframe, height=10, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        logscroll = ttk.Scrollbar(logframe, orient="vertical", command=self.log.yview)
        logscroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=logscroll.set)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert(tk.END, text)
        if not text.endswith("\n"):
            self.log.insert(tk.END, "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

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
                        f"Файл: {Path(raw_path).name}\n"
                        f"Тензоров: {info['tensor_count']}\n"
                        f"Слоёв LoRA: {info['module_count']}\n"
                        f"Ранги: {info['ranks']}\n"
                        f"Типы данных: {', '.join(info['dtypes']) or 'не определены'}\n\n"
                        "Предупреждения:\n• " + "\n• ".join(warnings) +
                        "\n\nДобавить файл несмотря на предупреждения?"
                    )
                    if not messagebox.askyesno("LoRA проверена с предупреждениями", details, parent=self.root):
                        rejected.append(Path(raw_path).name + " (отклонена пользователем)")
                        continue
                self.input_files.append(raw_path)
                self.weight_vars.append(tk.StringVar(value="1.0"))
                self.header_infos.append(info)
                added.append(Path(raw_path).name)
            except Exception as exc:
                rejected.append(Path(raw_path).name + f" — {exc}")
                messagebox.showerror(
                    "LoRA не прошла проверку",
                    f"Файл не добавлен в список:\n{Path(raw_path).name}\n\n{exc}\n\n"
                    "Проверка читает только заголовок SafeTensors. Исходный файл не изменяется.",
                    parent=self.root,
                )
        self.refresh_inputs()
        if added:
            self.status.set(f"Добавлено после проверки: {len(added)}")
        elif not paths:
            return
        elif rejected:
            self.status.set(f"Не добавлено: {len(rejected)}")

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

    def validate_and_build_args(self):
        if not self.input_files:
            raise ValueError("Add at least one input LoRA file.")
        for path in self.input_files:
            if not Path(path).is_file():
                raise ValueError(f"Input file does not exist:\\n{path}")

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
        import math
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
        )

    def start_merge(self):
        try:
            args = self.validate_and_build_args()
        except Exception as exc:
            messagebox.showerror("Check settings", str(exc), parent=self.root)
            return
        self.run_button.configure(state="disabled")
        self.status.set("Running…")
        self.append_log("\\n" + "=" * 70)
        self.append_log("Running merger directly inside this application (no subprocess).")
        self.append_log(f"Inputs: {len(args.inputs)}; rank={args.target_rank}; dtype={args.dtype}; dry_run={args.dry_run}")
        threading.Thread(target=self.run_worker, args=(args,), daemon=True).start()

    def run_worker(self, args):
        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                merge_loras(args)
            self.root.after(0, self.finish_run, 0, capture.getvalue())
        except Exception:
            output = capture.getvalue()
            output += "\\n--- ERROR ---\\n" + traceback.format_exc()
            self.root.after(0, self.finish_run, 1, output)

    def finish_run(self, returncode, output):
        if output.strip():
            self.append_log(output.rstrip())
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


def main():
    root = tk.Tk()
    Krea2MergerGUI(root)
    root.mainloop()

def launch_gui():
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
        try:
            merge_loras(args_parse())
            print("\\nFinished.")
        except Exception as exc:
            print(f"\\nERROR: {exc}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        finally:
            if sys.stdin and sys.stdin.isatty():
                try:
                    input("\\nPress Enter to close...")
                except EOFError:
                    pass
