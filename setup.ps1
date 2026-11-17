# VeriVault Forensics — Windows Quick Setup
# Run this script from PowerShell inside the cloned repo folder.
# Usage: .\setup.ps1

Write-Host "=== VeriVault Forensics Setup ===" -ForegroundColor Cyan

# 1. Create venv
Write-Host "`n[1/5] Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) { Write-Error "Python not found. Install Python 3.10+ first."; exit 1 }

# 2. Activate venv
Write-Host "[2/5] Activating venv..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# 3. Install PyTorch CPU-only (must come before requirements.txt)
Write-Host "[3/5] Installing PyTorch (CPU)..." -ForegroundColor Yellow
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
Write-Host "[4/5] Installing other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# 5. Force numpy<2 (EasyOCR/opencv-python can upgrade it; this breaks cv2)
Write-Host "[5/5] Pinning numpy to 1.26.4..." -ForegroundColor Yellow
pip install "numpy==1.26.4"

# 6. Pull LFS model file
Write-Host "`n[LFS] Pulling model weights (52 MB)..." -ForegroundColor Yellow
git lfs pull

Write-Host "`n=== Setup complete! ===" -ForegroundColor Green
Write-Host "Run the app with:  python app.py" -ForegroundColor Green
Write-Host "Then open:         http://localhost:5000" -ForegroundColor Green
