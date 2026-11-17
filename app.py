import os
import json
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from services.ocr_service import OCRService
from services.field_extractor import extract_fields
from services.aadhaar_validator import validate_aadhaar
from services.tamper_detector import detect_tamper
from services.ai_detector_service import AIDetectorService

app = Flask(__name__)
app.secret_key = 'super_secret_academic_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR and AI Forensics services
ocr_service = OCRService()
ai_detector_service = AIDetectorService()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------------
# Home & Aadhaar Routes (100% Preserved Existing Functionality)
# -------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_document():
    if 'document' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['document']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
        
    if not allowed_file(file.filename):
        flash('Unsupported file format. Please upload JPG, JPEG, PNG, or WEBP.')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 1. OCR text extraction
        ocr_text = ocr_service.extract_text(filepath)
        
        # 2. Field extraction
        fields = extract_fields(ocr_text)
        
        # 3. Aadhaar Validation
        validation_status = "Not Applicable"
        if fields["Document Type"] == "Aadhaar" and fields["Document Number"] != "Not Detected":
            is_valid = validate_aadhaar(fields["Document Number"])
            validation_status = "Basic validation passed" if is_valid else "Validation Failed"
            
            # Mask Aadhaar Number for display
            num = fields["Document Number"]
            if len(num) == 12:
                fields["Document Number"] = f"XXXX XXXX {num[-4:]}"
                
        # 4. Tamper Detection
        tamper_status, tamper_msg = detect_tamper(filepath)
        
        # 5. Final Decision Logic
        overall_status = "INVALID"
        if fields["Document Type"] == "Aadhaar":
            if validation_status == "Basic validation passed" and tamper_status == "CLEAN":
                overall_status = "VERIFIED"
            elif tamper_status == "POTENTIALLY SUSPICIOUS" or fields["Name"] == "Not Detected":
                overall_status = "REVIEW REQUIRED"
        elif fields["Document Type"] != "Not Detected":
            overall_status = "REVIEW REQUIRED"
        else:
            overall_status = "INVALID"

        return render_template(
            'results.html', 
            ocr_text=ocr_text,
            fields=fields,
            validation_status=validation_status,
            tamper_status=tamper_status,
            tamper_msg=tamper_msg,
            overall_status=overall_status
        )

# -------------------------------------------------------------
# New Feature: Universal AI Image & Manipulation Detector
# -------------------------------------------------------------

@app.route('/ai-detector', methods=['GET'])
def ai_detector_page():
    """Renders the AI Image & Manipulation Detector with Metrics Dashboard."""
    metrics_path = os.path.join('static', 'metrics', 'metrics_report.json')
    metrics_data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error loading metrics report: {e}")
            
    return render_template('ai_detector.html', metrics=metrics_data)

@app.route('/api/detect-ai-image', methods=['POST'])
def api_detect_ai_image():
    """
    Dedicated API endpoint for Deep Learning AI Image & Manipulation Detection.
    Accepts:
      - Multipart file upload under key 'image'
      - OR JSON payload with 'sample_name': 'sample_real' | 'sample_ai'
    Returns:
      JSON with verdict, confidence, probabilities, Grad-CAM base64, ELA base64, and forensic report.
    """
    try:
        filepath = None
        # Check if sample requested
        sample_name = request.form.get('sample_name')
        if not sample_name and request.is_json:
            sample_name = request.json.get('sample_name')
            
        if sample_name:
            sample_file = f"{sample_name}.jpg" if not sample_name.endswith('.jpg') else sample_name
            sample_path = os.path.join('static', 'samples', sample_file)
            if os.path.exists(sample_path):
                filepath = sample_path
            else:
                return jsonify({'error': f'Sample image {sample_file} not found'}), 404
        else:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided in request'}), 400
                
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
                
            if not allowed_file(file.filename):
                return jsonify({'error': 'Unsupported file format. Please upload JPG, PNG, or WEBP'}), 400
                
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"ai_check_{filename}")
            file.save(filepath)

        # Run deep learning forensic inference with Grad-CAM and ELA
        results = ai_detector_service.predict(filepath)
        return jsonify({
            'success': True,
            **results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Forensic analysis failed: {str(e)}'}), 500

@app.route('/api/model-metrics', methods=['GET'])
def api_model_metrics():
    """Serves the latest model training and evaluation metrics report."""
    metrics_path = os.path.join('static', 'metrics', 'metrics_report.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({'error': 'Metrics report not found. Train the model first.'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
