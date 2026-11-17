#!/bin/bash
# VeriVault Forensics — Mac/Linux Quick Setup
# Usage: bash setup.sh

set -e
echo "=== VeriVault Forensics Setup ==="

echo "[1/5] Creating virtual environment..."
python3 -m venv venv

echo "[2/5] Activating venv..."
source venv/bin/activate

echo "[3/5] Installing PyTorch (CPU)..."
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

echo "[4/5] Installing other dependencies..."
pip install -r requirements.txt

echo "[5/5] Pinning numpy to 1.26.4..."
pip install "numpy==1.26.4"

echo "[LFS] Pulling model weights (52 MB)..."
git lfs pull

echo ""
echo "=== Setup complete! ==="
echo "Run:  python app.py"
echo "Open: http://localhost:5000"
