#!/usr/bin/env bash
# Cloud GPU setup for the WiLoR mesh track. Python 3.10 is the version WiLoR-mini tests.
# HaMeR is the heavier alternative (separate detector + MANO download). This script
# installs WiLoR, which already returns 21 joints and a MANO mesh in one pass.
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install "git+https://github.com/warmshao/WiLoR-mini"
echo "WiLoR ready. First predict() downloads weights from Hugging Face."
