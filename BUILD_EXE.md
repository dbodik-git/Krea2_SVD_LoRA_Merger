# Build a standalone Windows EXE

The repository includes `build_exe.bat` to package the GUI merger as a single
Windows executable with PyInstaller.

## Before building

- Windows 10/11, 64-bit.
- Python and the project's runtime dependencies installed in the active environment:
  `torch`, `safetensors`, and `ttkbootstrap`.
- The build uses the Python environment selected by `python` on PATH.
- This merger performs its calculations on the CPU. A CPU-only PyTorch build is
  sufficient and generally makes a smaller bundle than a CUDA-enabled build.

Avoid building from Forge Neo's Python environment if you do not want PyInstaller
or build dependencies installed into it. Prefer a separate virtual environment
with a Python version supported by the installed PyTorch and PyInstaller.

## Build

1. Download/clone the repository and open its root folder.
2. Ensure the intended Python environment is active and `python` resolves to it.
3. Double-click `build_exe.bat`, or run it from Command Prompt.
4. Wait for PyInstaller to finish. The output is:

   `dist\Krea2_SVD_LoRA_Merger.exe`

The first build can take several minutes and the EXE can be large because PyTorch
and its dependencies are bundled. PyInstaller is installed automatically by the
batch file if it is missing.

## Test the EXE

- Launch it on the build computer and test GUI startup.
- Perform a small merge with test LoRAs and verify the output SafeTensors and
  Forge Neo sidecar JSON.
- Keep the source script and input files. The EXE does not include LoRA models,
  checkpoints, or user data.
- For a true “Python not installed” test, copy the EXE to a clean Windows
  machine/VM that has no Python installation.

## Notes

- This is a Windows build; it does not create macOS or Linux executables.
- `--onefile` extracts bundled files to a temporary directory at startup, so
  startup can be slower than the script or an `--onedir` build.
- Antivirus may take extra time to scan a newly built PyInstaller executable.
- A successful PyInstaller build is not by itself an integration test. Confirm
  the application actually starts and completes a merge on the target computer.
