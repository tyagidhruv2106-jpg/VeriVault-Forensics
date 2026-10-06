import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from services.ocr_service import OCRService
from services.field_extractor import extract_fields
from services.aadhaar_validator import validate_aadhaar
from services.tamper_detector import detect_tamper

app = Flask(__name__)
app.secret_key = 'super_secret_academic_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR (lazy loaded or globally)
ocr_service = OCRService()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        flash('Unsupported file format. Please upload JPG, JPEG, or PNG.')
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
