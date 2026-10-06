# VeriVault

## Project Overview
"VeriVault – AI-Powered Identity & Document Verification System" is an academic prototype demonstrating how Artificial Intelligence, Optical Character Recognition (OCR), and image forensics can be integrated to create an automated identity document screening system.

## Problem Statement
Manual verification of identity documents is time-consuming, prone to human error, and struggles to quickly identify tampered or digitally altered images.

## Existing System Limitations
- **Manual Effort:** Requires human agents to read and verify details.
- **Time Consumption:** Bottlenecks during onboarding processes.
- **Human Error:** Mistakes in transcription or missing anomalies.
- **Lack of automated preliminary screening:** No initial automated pass to flag obvious manipulations.

## Proposed Solution
VeriVault automates the preliminary screening of documents. It uses AI to read the document, extracts key identity fields, performs structural validation (e.g., checksum validation), and conducts a basic forensic analysis of the image to flag potential tampering.

## Objectives
- Automate preliminary document screening.
- Extract identity information using OCR.
- Perform basic identity-field validation.
- Detect potentially suspicious document manipulation.
- Provide a clear verification result dashboard.
- Build a foundation for future biometric and blockchain modules.

## Methodology
The system follows a linear pipeline:
1. **Document Upload** (Web Interface accepts JPG/PNG)
2. **Preprocessing & OCR** (EasyOCR extracts text)
3. **Field Extraction** (Regex-based logic identifies Name, DOB, Gender, ID Number)
4. **Validation** (Verhoeff algorithm checks Aadhaar format)
5. **Tamper Analysis** (Error Level Analysis checks for compression anomalies)
6. **Result Presentation** (Dashboard categorizes as VERIFIED, REVIEW REQUIRED, or INVALID)

## Technology Stack
- **Backend:** Python, Flask
- **AI / Computer Vision:** EasyOCR, OpenCV, Pillow, Numpy
- **Frontend:** HTML, CSS, JavaScript

## Current Implementation
- **OCR Module:** `services/ocr_service.py`
- **Field Extractor:** `services/field_extractor.py`
- **Validation Module:** `services/aadhaar_validator.py`
- **Tamper Detection:** `services/tamper_detector.py`
- **Frontend & App Engine:** `app.py`, `templates/`, `static/`

## Future Scope (Planned Modules)
- Face Verification (Matching ID photo with selfie)
- Liveness Detection (Ensuring user presence)
- Blockchain Audit Trail (Immutable logging of verification)

## Installation & How to Run
```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000` in your web browser.

## Limitations
This system is an academic prototype intended for *preliminary screening*. It does not replace official governmental verification platforms. The tamper detection (ELA) serves as a demonstration of image forensics and cannot guarantee a document's absolute authenticity.

## Conclusion
VeriVault successfully demonstrates the synchronization of project design and implementation for an AI-powered pipeline, offering a solid proof of concept that fulfills the required academic criteria.
