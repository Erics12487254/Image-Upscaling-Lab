# Image Super Resolution

## Purpose

This project grew out of an experiment lab for the [Ai-Enhancer-Toolkit](https://github.com/Erics12487254/Ai-Enhancer-Toolkit): a space for trying image-processing and machine-learning ideas hands-on. It explores image upscaling by comparing Bicubic, ESPCN, and Real-ESRGAN, and measuring their results with PSNR, SSIM, and runtime. The goal is to learn through practical experiments and make the comparisons easy to inspect.

Run the three x2 upscalers on one image per operation. The terminal asks for an image path, creates a low-resolution copy, writes the Bicubic, ESPCN, and Real-ESRGAN outputs, calculates PSNR and SSIM against the selected image, and stores the three results in `evaluation.db`. Each operation clears earlier evaluation rows.

## Setup

Use Python 3.10 or newer. From PowerShell in this folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Ensure the model files are at `models/espcn_color_x2.pth` and `models/RealESRGAN_x2plus.pth`.

## Run from the editor

Open `main/cli.py` and use **Run Code**. The workspace setting runs it in the terminal with the project's `.venv`, so you can type the requested image path. Running the file with no arguments starts the one-image workflow directly.

You can also run it from the project terminal:

```powershell
.venv\Scripts\python.exe main\cli.py
```

or use the module form:

```powershell
py -u -m main.cli run-pipeline
```

Enter a full image path when prompted. One image is processed per operation. Outputs are written to `data/low_resolution/` and `data/output/`; `evaluation.db` contains only the latest operation.
