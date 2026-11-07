
# Krea2 SVD LoRA Merger

Merge several LoRA files into one and re-pack the result to a uniform rank with SVD. Runs entirely on the CPU, never touches your source files, and comes with a small GUI and a command-line mode.

Typical use: you have three Krea2 LoRAs (a style, a character, a detail LoRA) and want a single `.safetensors` file with fixed coefficients, instead of stacking them at generation time.

## Features

- **Weighted merge** of any number of LoRAs, with a separate coefficient for each (negative values are allowed).
- **Uniform output rank** via QR + small-core SVD. Only a small `k x k` matrix goes through SVD, so it is fast even for large layers.
- **CPU only.** No GPU or VRAM needed.
- **Preflight checks** when you add a file: SafeTensors structure, A/B shapes, key naming, dimension compatibility with the other selected LoRAs. Only headers are read, so this is instant.
- **Safe by design:**
  - inputs are read-only and can never be chosen as the output;
  - unknown tensors stop the run instead of being silently dropped;
  - the result is written to a temp file, re-opened and checked for shape and NaN/Inf, and only then atomically moved into place;
  - dry-run mode validates everything without writing.
- **Reports the truncation error** (mean / max relative Frobenius error per layer) so you can judge whether the target rank is high enough.
- **Forge Neo user metadata JSON:** reads each input LoRA's sidecar `.json`, combines `activation text` and `negative text` phrases (case-insensitive deduplication), averages available `preferred weight` values, and writes a Forge-compatible six-field JSON beside the merged output. Missing input JSON files are skipped. Existing output JSON is protected unless `--overwrite` is enabled.
- **Modern GUI** built on [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) with switchable themes, live log and progress indicator. Falls back to plain `tkinter.ttk` if ttkbootstrap is not installed.
  
<img width="900" height="760" alt="Screen-2026-11-04_16-32-01" src="https://github.com/user-attachments/assets/8152c06a-f57a-436d-aa27-84658f6d5421" />

## Requirements

- Python 3.10 or newer (recommended; use a version supported by your installed PyTorch build)
- [`torch`](https://pytorch.org/) (CPU build is sufficient; install a PyTorch build compatible with your Python and platform)
- [`safetensors`](https://github.com/huggingface/safetensors)
- GUI: `tkinter` (usually included with Windows Python installers); `ttkbootstrap` is optional and only provides the themed GUI

```bash
python -m pip install -r requirements.txt
```

The command-line mode works without `tkinter` and `ttkbootstrap`. The included `requirements.txt` pins a full dependency set and includes `ttkbootstrap`; if you only need CLI, install `torch` and `safetensors` compatible with your environment instead. Avoid installing these packages into Forge Neo's Python unless you intentionally want to modify that environment.

## Usage

### GUI

Run the script without arguments:

```bash
python Krea2_SVD_LoRA_Merger.py
```

1. Click **Add files...** and pick your LoRAs. Each file is checked on the spot; problems are explained in a dialog.
2. Set a **coefficient** for every LoRA (default `1.0`).
3. Choose the **output file**, **target rank** and **dtype**.
4. Optionally tick **Dry run** to validate without saving.
5. Click **Run merger** and watch the log.
6. After a successful merge, a matching Forge Neo `.json` sidecar is also saved. Its activation/negative phrases are gathered from the input sidecar JSON files. In the GUI, the **Forge Neo activation text** box is prefilled from those files and can be edited manually before merging; the edited text is what gets saved.

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
| `--output` | example path in the script | Output file |
| `--weights` | `1.0` for every input | One coefficient per input |
| `--target-rank` | `32` | Rank of the merged LoRA |
| `--dtype` | `bf16` | Output dtype: `bf16`, `fp16` or `fp32` |
| `--dry-run` | off | Validate everything, write nothing |
| `--overwrite` | off | Allow replacing an existing output file |
| `--no-pause` | off | Do not wait for Enter after an error (for scripts and `.bat` files) |

The default folder `D` is set to `E:\SD\Lora\krea2` for the author's Windows setup. The `lora1.safetensors` etc. names are placeholders: edit `DEFAULT_INPUTS` and `DEFAULT_OUTPUT` to your own filenames or always pass `--inputs` and `--output`.

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
- Alpha keys: `.alpha` or `.lora_alpha`. Missing alpha is treated as `alpha = rank` (scale 1).
- Linear (2D) layers only.
- Layer names must match between LoRAs to be summed. A layer that exists in only one LoRA is kept and re-packed on its own.
- The output reuses the key style of the first LoRA that contains the layer.

## Limitations and notes

- **Conv (4D) LoRAs, LoKr and LoHa are not supported** and are rejected with an error.
- **Alpha without tensors:** if a LoRA has no alpha tensors but was trained with `alpha != rank` (the value is then only in `ss_network_alpha` metadata or in a training config), the script assumes scale 1 and prints a warning. Check such files by testing the result.
- **Memory:** all input LoRAs are loaded into RAM at once, in their stored dtype.
- **Metadata:** the output is labeled `modelspec.architecture = krea2/lora` and `ss_base_model_version = krea2`. Along with it the file records the merge inputs, coefficients, target rank and dtype (`merge_*` keys). Change these in `merge_loras()` if you use the script for another model.
- Always test a merged LoRA at low strength first; merged deltas can behave differently from stacked LoRAs, especially at high coefficients.

## Troubleshooting

| Problem | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: tkinter` | Your Python has no Tk. Use the command-line mode, or install Python with Tcl/Tk support. |
| GUI looks plain | `ttkbootstrap` is not installed: `pip install ttkbootstrap`. |
| `Unrecognized tensors ...` | The file contains keys that are not LoRA A/B/alpha pairs (for example LoKr/LoHa or extra data). The script stops on purpose rather than dropping data. |
| `Dimension mismatch ...` | The LoRAs were trained for different base models or architectures. |
| `fp16 overflow at ...` | Use `bf16` or `fp32` as the output dtype. |
| `Output exists ...` | Pick another file name or pass `--overwrite` (or tick the checkbox in the GUI). |

