# WiLoR mesh track on a local NVIDIA GPU (Windows).
# RTX 3050 6GB: use fp16, --max-frames for demos, close other GPU apps first.
# Requires Python 3.10 (recommended). Create a separate venv if your main env is 3.11+.
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "Checking NVIDIA driver..."
nvidia-smi | Out-Host

& $Python -m pip install --upgrade pip
& $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
& $Python -m pip install "git+https://github.com/warmshao/WiLoR-mini"

& $Python -c @"
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f'vram_free_mb {free // (1024*1024)} / {total // (1024*1024)}')
"@

Write-Host @"

WiLoR installed. Example (6GB GPU — start small):

  `$Python scripts/run_mesh.py --input data/raw/clip.mp4 --max-frames 120 --device cuda --fp32
  # If OOM, omit --fp32 (use fp16) or add --stride 2

Merge onto MediaPipe JSONL:

  `$Python scripts/run_mesh.py --input data/raw/clip.mp4 --merge outputs/clip.jsonl
"@
