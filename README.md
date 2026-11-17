# VeriVault: AI Image Forensics & Multi-Layer Identity Verification Platform

VeriVault is a complete, production-grade image forensics and document authenticity verification platform. It integrates mathematical checksum validation, optical character recognition (OCR), Error Level Analysis (ELA), and deep transfer learning (EfficientNet-B0) with Grad-CAM explainability to detect AI-generated imagery and document manipulation.

---

## System Architecture & Features

```
                                      VERIVAULT PLATFORM
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             │                                                                   │
             ▼                                                                   ▼
┌───────────────────────────────┐                               ┌─────────────────────────────────┐
│     MODULE 1: AADHAAR CHECK   │                               │  MODULE 2: UNIVERSAL AI DETECT  │
│  (Existing Unchanged Engine)  │                               │    (Deep Learning Forensics)    │
├───────────────────────────────┤                               ├─────────────────────────────────┤
│ • EasyOCR (English & Hindi)   │                               │ • Dual-Stream EfficientNet-B0   │
│ • Verhoeff D8 Checksum Algo   │                               │ • 6-Channel (RGB + ELA Residual)│
│ • JPEG Quality Difference ELA │                               │ • Grad-CAM Attention Heatmaps   │
│ • Redacted Storage & Review   │                               │ • Academic Metrics Dashboard    │
└───────────────────────────────┘                               └─────────────────────────────────┘
```

### Module 1: Aadhaar Card Authenticity Check (Intact & Untouched)
- **Mathematical Checksum**: Implements the official Verhoeff dihedral group $D_8$ permutation algorithm to validate 12-digit Indian Aadhaar numbers.
- **Dual-Language Text Extraction**: Leverages EasyOCR to extract English and Hindi demographic data (Name, DOB, Gender, UID).
- **Forensic Compression Analysis**: Basic JPEG Error Level Analysis (ELA) flags suspicious re-compression anomalies.
- **Privacy Masking**: Automatically masks sensitive Aadhaar digits for storage and display compliance (`XXXX XXXX 1234`).

### Module 2: Universal AI Image & Manipulation Detector
- **Dual-Stream Forensic Architecture**: Combines a 3-channel RGB image with a 3-channel Error Level Analysis (ELA) residual map into a 6-channel convolutional stem, preserving pretrained ImageNet weights for visual semantics while exposing frequency/quantization discrepancies.
- **Deep Neural Backbone**: Transfer learning built on **EfficientNet-B0** with a custom classification head featuring Dropout ($p=0.4$), Batch Normalization, and SiLU activations.
- **Grad-CAM Explainability**: Generates gradient-weighted class activation heatmaps overlaying the original image to expose which spatial regions and generative artifacts triggered the prediction.
- **Academic Evaluation Dashboard**: Live browser inspection of model metrics:
  - Classification Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  - Training vs. Validation Loss & Accuracy curves (`training_curves.png`).
  - Confusion Matrix on held-out test data (`confusion_matrix.png`).
  - Receiver Operating Characteristic (ROC) curve (`roc_curve.png`).
  - Complete per-epoch numeric history and architecture specifications.

---

## Directory Structure

```
VeriVault/
├── app.py                         # Flask backend application & API routes
├── requirements.txt               # Pinned Python package dependencies
├── test_app_integration.py        # Automated test suite
├── models/
│   └── ai_detector_best.pth       # Trained PyTorch checkpoint
├── ml_pipeline/
│   ├── prepare_dataset.py         # Multi-dataset ingestion & benchmark generator
│   └── train_model.py             # EfficientNet-B0 training & artifact generator
├── services/
│   ├── aadhaar_validator.py       # Verhoeff algorithm logic
│   ├── field_extractor.py         # Regex & field parsing logic
│   ├── ocr_service.py             # EasyOCR text extractor
│   ├── tamper_detector.py         # Basic ELA tamper detector
│   └── ai_detector_service.py     # AI inference, Grad-CAM, & ELA service
├── static/
│   ├── css/style.css              # Cyber-forensics dark mode styles
│   ├── js/
│   │   ├── script.js              # Aadhaar page drag-and-drop controller
│   │   └── ai_detector.js         # AI detector & metrics tab controller
│   ├── metrics/                   # Generated evaluation artifacts
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── metrics_report.json
│   └── samples/                   # Built-in evaluation test images
│       ├── sample_real.jpg
│       └── sample_ai.jpg
├── templates/
│   ├── index.html                 # Dual-card Home Page & Aadhaar scanner
│   ├── results.html               # Aadhaar verification dashboard
│   └── ai_detector.html           # Universal AI detector & metrics tab
└── dataset/                       # Standardized forensic dataset splits
    ├── train/ (real/ & ai_manipulated/)
    ├── val/   (real/ & ai_manipulated/)
    └── test/  (real/ & ai_manipulated/)
```

---

## Step 1: Environment Setup

Clone or open the repository in terminal and set up the Python virtual environment:

```bash
# 1. Create virtual environment (Python 3.10, 3.11, or 3.12)
python -m venv venv

# 2. Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
.\venv\Scripts\activate.bat
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Dataset Setup & Preprocessing

The preprocessing pipeline (`ml_pipeline/prepare_dataset.py`) standardizes images to `224x224` RGB format and splits them into 70% Train, 15% Validation, and 15% Test sets.

### Option A: Instant Benchmark Generation (Single Command Out-of-the-Box)
Run the automated synthesizer to instantly produce realistic photographic images vs. AI generative frequency anomalies and CASIA-style localized splicing:

```bash
python ml_pipeline/prepare_dataset.py --generate-sample-dataset --samples-per-class 100
```

### Option B: Ingesting Real-World Datasets (Tiny-GenImage & CASIA 2.0 / CocoGlide)
Download Tiny-GenImage and CASIA 2.0 datasets and ingest them directly:

```bash
# Ingest Tiny-GenImage (Stable Diffusion, Midjourney, GANs, ADM, Glide)
python ml_pipeline/prepare_dataset.py --source-tinygenimage /path/to/Tiny-GenImage

# Ingest CASIA 2.0 (Splicing & Localized Inpainting)
python ml_pipeline/prepare_dataset.py --source-casia /path/to/CASIA2.0

# Combine both datasets simultaneously into unified train/val/test splits:
python ml_pipeline/prepare_dataset.py --source-tinygenimage /path/to/Tiny-GenImage --source-casia /path/to/CASIA2.0
```

### Downloading via Kaggle CLI
If you have Kaggle API configured (`~/.kaggle/kaggle.json`):
```bash
kaggle datasets download -d sophiaz/casia-dataset -p ./raw_data/casia --unzip
kaggle datasets download -d birdy654/deepfake-faces -p ./raw_data/genimage --unzip
```

---

## Step 3: Model Architecture & Training

Train the dual-stream EfficientNet-B0 forensic classifier:

```bash
python ml_pipeline/train_model.py --epochs 8 --batch-size 16 --lr 0.0003
```

### Arguments:
- `--epochs`: Number of training epochs (default: `8`).
- `--batch-size`: Batch size for DataLoaders (default: `16`).
- `--lr`: Learning rate for AdamW optimizer (default: `3e-4`).
- `--data-dir`: Dataset root directory (default: `dataset`).
- `--output-model`: Path for best checkpoint (default: `models/ai_detector_best.pth`).
- `--metrics-dir`: Path for evaluation graphs and JSON (default: `static/metrics`).

### Automatic Outputs Generated:
1. `models/ai_detector_best.pth` — Best model weights saved via early stopping.
2. `static/metrics/training_curves.png` — Side-by-side Loss and Accuracy curves.
3. `static/metrics/confusion_matrix.png` — Annotated test set confusion matrix.
4. `static/metrics/roc_curve.png` — ROC curve with computed ROC-AUC score.
5. `static/metrics/metrics_report.json` — Numeric metrics report for the UI dashboard.

---

## Step 4: Launching the Web Application

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
- **Home Page**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **Universal AI Detector & Metrics Dashboard**: [http://127.0.0.1:5000/ai-detector](http://127.0.0.1:5000/ai-detector)
- **Model Performance Tab**: [http://127.0.0.1:5000/ai-detector#metrics](http://127.0.0.1:5000/ai-detector#metrics)

---

## REST API Reference

### 1. AI Image & Manipulation Detection API
- **Endpoint**: `POST /api/detect-ai-image`
- **Payload**:
  - Multipart form data with file field `image`: `(binary image)`
  - OR JSON body with sample selection: `{"sample_name": "sample_real"}` or `{"sample_name": "sample_ai"}`
- **Response**:
```json
{
  "success": true,
  "verdict": "AI-Generated / AI-Edited",
  "verdict_badge": "AI_MANIPULATED",
  "confidence_score": 93.7,
  "prob_real": 6.3,
  "prob_ai": 93.7,
  "dimensions": "224 x 224",
  "original_image": "data:image/png;base64,...",
  "gradcam_image": "data:image/png;base64,...",
  "ela_image": "data:image/png;base64,...",
  "forensic_explanation": [
    "Convolutional attention localized high-frequency generative artifacts with 93.7% confidence.",
    "Error Level Analysis (ELA) reveals non-uniform compression residuals and quantization boundary discrepancies typical of latent diffusion or localized splicing.",
    "Spectral deconvolution signatures and spatial frequency anomalies detected in the attention focal areas highlighted by the red/warm Grad-CAM hotspots."
  ]
}
```

### 2. Model Performance Metrics API
- **Endpoint**: `GET /api/model-metrics`
- **Response**: Serves `static/metrics/metrics_report.json` containing exact test accuracy, precision, recall, F1, ROC-AUC, and per-epoch arrays.

### 3. Aadhaar Verification (Preserved Legacy Route)
- **Endpoint**: `POST /scan`
- **Payload**: Multipart form data with `document`: `(Aadhaar image file)`
- **Response**: Returns rendered `results.html` dashboard with OCR extracted fields, Verhoeff checksum result, and tamper status.

---

## Automated Test Verification

Run the integration test suite to verify all modules and endpoints:

```bash
python test_app_integration.py
```

