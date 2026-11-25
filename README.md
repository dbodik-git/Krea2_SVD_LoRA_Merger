
# Krea2 SVD LoRA Merger

Merge several LoRA files into one and re-pack the result to a uniform rank with SVD. Runs entirely on the CPU, never touches your source files, and comes with a small GUI and a command-line mode.

Typical use: you have three Krea2 LoRAs (a style, a character, a detail LoRA) and want a single `.safetensors` file with fixed coefficients, instead of stacking them at generation time.

## Features

- **Weighted merge** of any number of LoRAs, with a separate coefficient for each (negative values are allowed).
- **Uniform output rank** via QR + small-core SVD. Only a small `k x k` matrix goes through SVD, so it is fast even for large layers.
- **CPU only.** No GPU or VRAM needed.
- **CPU thread control:** choose 1–20 PyTorch intra-op threads in CLI or GUI; the default is 6. More threads are not automatically faster, so benchmark on the target CPU.
- **Run diagnostics:** each run reports elapsed time, start/end working set, the peak observed by a 50 ms sampler, and the OS-reported lifetime peak working set when available. Working set includes shared and private resident pages (including mapped-file pages); it is process memory, not total system RAM. The sampler can miss brief peaks.
- **Preflight checks** when you add a file: SafeTensors structure, A/B shapes, key naming, dimension compatibility with the other selected LoRAs. Only headers are read, so this is instant.
- **Safe by design:**
  - inputs are read-only and can never be chosen as the output;
  - unknown tensors stop the run instead of being silently dropped;
  - the result is written to a temp file, re-opened and checked for shape and NaN/Inf, and only then atomically moved into place;
  - dry-run mode validates everything without writing.
- **Reports the truncation error** (mean / max relative Frobenius error per layer) so you can judge whether the target rank is high enough.
- **Forge Neo user metadata JSON:** reads each input LoRA's sidecar `.json`, combines `activation text` and `negative text` phrases (case-insensitive deduplication), averages available `preferred weight` values, and writes a Forge-compatible six-field JSON beside the merged output. Missing input JSON files are skipped. Existing output JSON is protected unless `--overwrite` is enabled.
- **Modern GUI** built on [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) with switchable themes, live log and progress indicator. Falls back to plain `tkinter.ttk` if ttkbootstrap is not installed.
  
<img width="918" height="1043" alt="Screen-2026-11-25_15-13-31" src="https://github.com/user-attachments/assets/c51c3b02-0705-41dd-9b33-5dab02291963" />

## Requirements

- Python 3.10 or newer (recommended; use a version supported by your installed PyTorch build)
- [`torch`](https://pytorch.org/) (CPU build is sufficient; install a PyTorch build compatible with your Python and platform)
- [`safetensors`](https://github.com/huggingface/safetensors)
- GUI: `tkinter` (usually included with Windows Python installers); `ttkbootstrap` is optional and only provides the themed GUI

```bash
python -m pip install -r requirements.txt
```

The command-line mode works without `tkinter` and `ttkbootstrap`. The included `requirements.txt` pins a full dependency set and includes `ttkbootstrap`; if you only need CLI, install `torch` and `safetensors` compatible with your environment instead. Avoid installing these packages into Forge Neo's Python unless you intentionally want to modify that environment.

The merger sets PyTorch's CPU intra-op thread count from `--cpu-threads` before tensor payloads are loaded. Each run prints elapsed wall time, start/end process working set, the peak observed by a 50 ms sampler, and the operating-system lifetime peak working set when available. On Windows, working set includes both shared and private resident pages, including mapped-file pages; it is not the same as private working set and is not total system RAM. The sampler can miss very short-lived peaks, while the Windows lifetime-peak counter records the process high-water mark.

## Usage

### GUI

Run the script without arguments:

```bash
python Krea2_SVD_LoRA_Merger.py
```

1. Click **Add files...** and pick your LoRAs. Each file is checked on the spot; problems are explained in a dialog.
2. Set a **coefficient** for every LoRA (default `1.0`).
3. Choose the **output file** (it must end in `.safetensors`), **target rank**, **dtype** and **CPU threads** (default 6; choose 1–20).
4. Check the **activation text** field. It is filled automatically from the input LoRAs' JSON files and you can edit it (see [Forge Neo sidecar JSON](#forge-neo-sidecar-json)).
5. Optionally tick **Dry run** to validate without saving, or untick **Write Forge Neo sidecar JSON** if you do not need it.
6. Click **Run merger** and watch the log. Click **Cancel** to stop a running merge (see [Cancelling](#cancelling)).

Use the theme selector in the bottom-right corner to change the look. The default theme is set by `DEFAULT_THEME` at the top of the script.

### Command line

```bash
python Krea2_SVD_LoRA_Merger.py \
    --inputs style.safetensors character.safetensors detail.safetensors \
    --weights 1.0 0.7 0.5 \
    --output merged_r32.safetensors \
    --target-rank 32 \
    --dtype bf16
```

On Windows `cmd`, put it on one line or use `^` instead of `\`.

| Option | Default | Description |
| --- | --- | --- |
| `--inputs` | three example paths in the script | Input `.safetensors` LoRA files |
| `--output` | example path in the script | Output file (must end in `.safetensors`) |
| `--weights` | `1.0` for every input | One coefficient per input |
| `--target-rank` | `32` | Rank of the merged LoRA |
| `--cpu-threads` | `6` | PyTorch CPU intra-op threads (1–20); more threads may be slower on some CPUs |
| `--lokr-full-rank` | off | Preserve the exact full LoKr factor rank until the final merge SVD (higher RAM use) |
| `--lokr-max-factor-gb` | `2.0` | Preflight limit in GiB for expanded full-rank LoKr A/B factors |
| `--lokr-matrix-free` | off | Use approximate matrix-free randomized joint SVD for LoKr + LoRA |
| `--lokr-svd-oversample` | `8` | Randomized SVD oversampling |
| `--lokr-svd-power-iterations` | `2` | Randomized SVD power iterations |
| `--lokr-workspace-max-gb` | `2.0` | Preflight limit in GiB for matrix-free algorithm workspace |
| `--dtype` | `bf16` | Output dtype: `bf16`, `fp16` or `fp32` |
| `--dry-run` | off | Validate everything, write nothing |
| `--overwrite` | off | Allow replacing an existing output file and its sidecar JSON (the old JSON is kept as `.json.bak`) |
| `--no-pause` | off | Do not wait for Enter after an error (for scripts and `.bat` files) |
| `--activation-text` | collected from the input JSON files | Activation text for the output Forge Neo JSON (an empty string `""` writes an empty field) |
| `--no-forge-json` | off | Do not write the Forge Neo sidecar JSON |

The default paths near the top of the script (`D`, `DEFAULT_INPUTS`, `DEFAULT_OUTPUT`) are examples for a Windows setup; edit them to match your folders or always pass `--inputs` and `--output`.

## Forge Neo sidecar JSON

Forge Neo reads per-LoRA user metadata from a JSON file with the same name as the LoRA (`my_lora.safetensors` is paired with `my_lora.json`). By default the merger writes this file next to the output, built from the JSON files that sit next to the input LoRAs. Inputs without a JSON file are simply skipped.

| Field | How it is built |
| --- | --- |
| `description` | `Merged LoRA: ` followed by the input file names |
| `sd version` | The most common value among the input JSON files (`Unknown` is ignored). If none has one, the `FORGE_SD_VERSION_FALLBACK` constant at the top of the script is used (`krea`). |
| `activation text` | Phrases from all inputs, comma-separated, duplicates removed (case-insensitive), or your own text |
| `preferred weight` | The average of the positive values in the inputs. Inputs without a value are ignored, and `0` (Forge's "not set") is written if there are none. |
| `negative text` | Merged the same way as the activation text |
| `notes` | `Merged from: ` followed by the input names, plus the notes of the inputs |

A few things to know:

- The merge coefficients are already baked into the weights, so the averaged `preferred weight` is only a starting point. Test the result and adjust it.
- **Activation text in the GUI:** the field is refilled whenever you add or remove files. Text you typed yourself is not overwritten; the **Reload from JSON** button rebuilds it. If you clear the field on purpose, an empty activation text is saved. From the command line use `--activation-text`.
- **Existing JSON:** without `--overwrite` (or the overwrite checkbox) an existing JSON stops the run before any work is done. With overwrite, the old file is first copied to `<name>.json.bak`, because it may contain hand-written notes.
- If the LoRA was saved but the JSON could not be written, you get a warning rather than a failed run.
- Use `--no-forge-json` or untick the checkbox in the GUI to skip the file. A dry run prints the JSON that would be written.

## Cancelling

The **Cancel** button in the GUI is active while a merge is running. The merge stops at the next safe point: before each input file is loaded, between layers, and right before saving. Once the output starts being written, saving and validation are not interrupted, which keeps the atomic-save guarantee. A cancelled run writes neither the LoRA nor the JSON. Closing the window during a run asks for confirmation and cancels in the same way.

In command-line mode, press `Ctrl+C`; temporary files are cleaned up.
## How it works

For every layer, each LoRA contributes a weight delta

```
delta_i = w_i * (alpha_i / rank_i) * B_i @ A_i
```

Summing the deltas is equivalent to stacking the factors:

```
A_cat = [A_1; A_2; ...]   (concatenated along the rank dimension)
B_cat = [w_1*s_1*B_1, w_2*s_2*B_2, ...]   (concatenated along the rank dimension, s_i = alpha_i / rank_i)
sum(delta_i) = B_cat @ A_cat
```

The stacked rank is the sum of all input ranks, so it is re-packed to the target rank:

1. `B_cat = Qb Rb` and `A_cat^T = Qa Ra` (reduced QR).
2. `B_cat A_cat = Qb (Rb Ra^T) Qa^T`, and only the small core `Rb Ra^T` goes through SVD.
3. The top `k` singular values are kept and `sqrt(S)` is split between the new `A` and `B`.
4. The output `alpha` equals the target rank, so the effective scale is exactly 1.

If the stacked rank is smaller than the target rank, the extra rank is zero-padded.

The reported per-layer metric is the relative Frobenius norm of the discarded singular-value tail: `||S[k:]||₂ / ||S||₂` (with zero returned when `||S||₂ = 0`). The summary reports the mean and maximum of this metric across layers. It estimates only rank-truncation loss; it does not include final output-dtype quantization error.

## Supported LoRA formats

- Key styles: `lora_down` / `lora_up` (kohya-style) and `lora_A` / `lora_B` (PEFT/diffusers-style, including `.default`).
- **LoKr input conversion:** direct full-matrix `lokr_w1` + `lokr_w2` pairs are converted internally to Kohya-style LoRA factors; the output then follows the same SVD merge path as ordinary LoRAs.
- Alpha keys: `.alpha` or `.lora_alpha`. Missing alpha is treated as `alpha = rank` (scale 1).
- Linear (2D) layers only.
- Layer names must match between LoRAs to be summed. A layer that exists in only one LoRA is kept and re-packed on its own.
- **One output key style for the whole file:** when ordinary LoRAs use different Kohya/PEFT suffix styles, the merger selects the most common style across all ordinary-LoRA modules (ties are resolved deterministically). The choice is made globally, so the output does not mix key styles and does not depend on input-file order. With LoKr-only inputs, Kohya-style keys are used.

## Limitations and notes

- **LoKr:** direct full-matrix `lokr_w1` / `lokr_w2` modules are detected automatically and converted to ordinary SVD LoRA at the selected target rank before merging. Conversion uses the factor SVDs and does not materialize the full Kronecker matrix. For direct full-matrix LyCORIS LoKr, alpha is not used as an extra scale; the converted LoRA is written with alpha equal to the target rank. Decomposed LoKr (`lokr_w*_a/b`), Tucker (`lokr_t2`) and LoHa are not supported and are rejected explicitly. Conv (4D) LoRA remains unsupported.
- **LoKr alpha semantics:** for direct full-matrix LoKr (`lokr_w1` + `lokr_w2`), the implementation applies `kron(w1, w2)` with scale 1; it reads and validates stored alpha but does not multiply by `alpha/rank`. This matches the documented LyCORIS full-matrix convention, but the trainer used for an arbitrary LoKr file may differ. The program therefore prints an explicit **UNVERIFIED** note about producer semantics. If that trainer scales full-matrix LoKr by `alpha/rank`, compensate with the corresponding merge coefficient. An alpha value matching `ss_network_dim` is consistent with the LyCORIS convention, but is not proof of the file's provenance.
- **Memory (standard mode):** all input LoRAs are loaded into RAM at once, in their stored dtype. The separate matrix-free mode reads only the current module's source tensors, but the output factors still accumulate in RAM.
- **Metadata:** when all inputs agree, the output preserves their compatible `modelspec.architecture` and base-model identity. Otherwise it falls back to `krea2/lora` and `krea2`. LoKr architecture labels are not copied to an ordinary LoRA output. The file also records the merge inputs, coefficients, target rank, dtype and error metric (`merge_*` keys).
- Always test a merged LoRA at low strength first; merged deltas can behave differently from stacked LoRAs, especially at high coefficients.

## Troubleshooting

| Problem | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: tkinter` | Your Python has no Tk. Use the command-line mode, or install Python with Tcl/Tk support. |
| GUI looks plain | `ttkbootstrap` is not installed: `pip install ttkbootstrap`. |
| `Unrecognized tensors ...` | The file contains unsupported or mixed tensor keys (for example LoHa, decomposed/Tucker LoKr, or extra data). Supported full-matrix LoKr is converted automatically. The script stops rather than dropping unknown data. |
| `Dimension mismatch ...` | The LoRAs were trained for different base models or architectures. |
| `fp16 overflow at ...` | Use `bf16` or `fp32` as the output dtype. |
| `Output exists ...` | Pick another file name or pass `--overwrite` (or tick the checkbox in the GUI). |



## Matrix-free joint LoKr + LoRA SVD (approximate)

The separate **Matrix-free joint SVD** mode combines direct-matrix LoKr terms and
ordinary LoRA terms through an implicit linear operator. It does not construct
the Kronecker matrix or expand LoKr into full-rank A/B factors. Modules are
processed one at a time, so input payloads are not all retained in RAM together.

This mode uses randomized SVD and is therefore approximate. Its reported
relative residual is estimated with random probe vectors; it is not the exact
Frobenius truncation error reported by the standard QR/SVD path. Negative merge
coefficients are supported for both LoKr and ordinary LoRA contributions.

CLI example:

~~~bash
python Krea2_SVD_LoRA_Merger.py --inputs lokr.safetensors style.safetensors ^
  --weights 1.0 -0.35 --output merged_matrix_free.safetensors ^
  --target-rank 32 --lokr-matrix-free --lokr-svd-oversample 8 ^
  --lokr-svd-power-iterations 2 --lokr-workspace-max-gb 2.0
~~~

The GUI has a separate checkbox for this mode. It cannot be combined with
**Preserve full LoKr rank until the final SVD**. The workspace preflight is a
conservative algorithm-workspace estimate only: it excludes input tensor
payloads, output tensors, and Python/PyTorch runtime memory. Keep the limit
conservative on systems with 16 GB RAM. Start with the default oversampling and
power-iteration settings; increasing them raises runtime and memory use.

The matrix-free mode currently supports the same direct full-matrix LoKr
layout as the existing converter (lokr_w1 + lokr_w2) and ordinary 2D LoRA
A/B pairs. Decomposed/Tucker LoKr, LoHa, and convolutional factors remain
unsupported.


## LoKr merge modes and error interpretation

The merger offers three ways to handle direct full-matrix LoKr together with ordinary LoRA:

| Mode | What happens | Rank truncation | Error information |
| --- | --- | --- | --- |
| Default | Each LoKr module is converted to ordinary A/B factors at the target rank, then all inputs are merged. | Two stages for LoKr: conversion, then final merge. | Exact conversion-stage LoKr truncation diagnostics, final-stage truncation error, and (when mathematically possible) a conservative upper bound for the combined error. |
| `--lokr-full-rank` | LoKr is converted to exact full-rank A/B factors before the joint merge. | One final merge truncation. | The final QR/SVD discarded-singular-value metric is exact for the represented sum. |
| `--lokr-matrix-free` | LoKr and LoRA are combined as an implicit linear operator; no full Kronecker matrix or expanded full-rank LoKr A/B factors are materialized. | One randomized joint approximation. | Random-probe relative residual estimate; approximate, not a formal bound or exact Frobenius error. |

The two explicit modes are mutually exclusive. The GUI can suggest/enable matrix-free mode after detecting a LoKr input; the user can decline or change the option. The full-rank mode is available as a separate checkbox and CLI option.

### Reading the error diagnostics

- **Default mode:** the final QR/SVD truncation metric is measured against the already-converted intermediate sum. It does not, by itself, include the earlier LoKr conversion loss. The merger also reports the conversion-stage loss and attempts to calculate a triangle-inequality **upper bound** for the total error relative to the exact weighted input sum. That bound can be loose; if its denominator cannot be shown to be positive, no bound is reported.
- **Full-rank mode:** there is no intermediate LoKr truncation, so the final discarded-singular-value metric covers the single truncation of the represented weighted sum (but not output-dtype quantization).
- **Matrix-free mode:** the reported residual is estimated using random probe vectors and includes randomized approximation plus truncation effects. It is a diagnostic estimate, not a formal upper bound and not guaranteed to equal the optimal rank-k error.
- A warning threshold such as 0.01 is a diagnostic hint, not a universal pass/fail quality standard.

### LoKr alpha: implementation convention and uncertainty

For direct full-matrix LoKr, the code forms the Kronecker update from `lokr_w1` and `lokr_w2` with scale 1. Alpha tensors are checked and recorded for diagnostics, but are not applied as an additional `alpha/rank` multiplier. This follows the LyCORIS full-matrix convention described in the source notes.

However, the exact trainer that produced a particular LoKr file may be unknown. Files with no alpha tensors (common in some PEFT-style exports) do not reveal the producer's intended scale in the SafeTensors file alone. The merger prints an **UNVERIFIED** note for this reason. If independent evidence shows that a producer applies `alpha/rank` to full-matrix LoKr, adjust that input's merge coefficient accordingly. Do not assume that every LoKr trainer uses the same convention.

### Memory and repeatability

- Standard/default mode loads all selected LoRA payloads into RAM.
- Matrix-free mode reads source tensors one module at a time, but output factors still accumulate in RAM. Its workspace preflight estimates algorithm workspace only; it excludes source payloads, output tensors, and Python/PyTorch runtime memory.
- Full-rank mode estimates the float32 storage needed for expanded LoKr A/B factors before loading payloads. This estimate does not include every temporary allocation or runtime overhead.
- Matrix-free randomized SVD uses a deterministic per-module seed derived from the module name, making the random stream stable across runs and input ordering for a given module. Floating-point/library differences can still affect bitwise reproducibility.
- The output key suffix style is selected once for the whole file using the most common ordinary-LoRA style, with deterministic tie-breaking. This prevents mixed key styles and makes the choice independent of input ordering.

## Building a standalone Windows executable

See [BUILD_EXE.md](BUILD_EXE.md) for the PyInstaller one-file Windows build instructions, environment requirements, output location, and smoke-test checklist. The executable bundles the application and its dependencies; it does not include LoRA models or user data.

## Exact LoKr preservation (single final SVD)

The optional **Preserve full LoKr rank until the final SVD** mode avoids the
intermediate rank truncation used by the default LoKr converter. Direct
full-matrix LoKr factors are expanded into exact ordinary A/B factors, then
combined with the other selected LoRAs and compressed once by the normal
joint QR + small-core SVD stage.

This mode can use substantially more RAM. Before loading tensor payloads, the
merger estimates the float32 A/B factor storage from the SafeTensors headers.
The GUI uses a conservative 2 GiB estimate limit. CLI example:

```bash
python Krea2_SVD_LoRA_Merger.py --inputs lokr_a.safetensors lokr_b.safetensors ordinary.safetensors --output merged_fullrank.safetensors --target-rank 32 --lokr-full-rank --lokr-max-factor-gb 2.0
```

The estimate covers expanded LoKr A/B factors, not every temporary tensor or
the Python/PyTorch runtime. Keep the limit conservative on systems with 16 GB
RAM; do not raise it unless sufficient physical memory is available. The
default mode remains the lower-memory per-LoKr rank conversion.
