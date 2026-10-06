import easyocr

class OCRService:
    def __init__(self):
        # Initialize reader for English and Hindi (using CPU for reliable compatibility)
        self.reader = easyocr.Reader(['en', 'hi'], gpu=False)

    def extract_text(self, image_path):
        try:
            results = self.reader.readtext(image_path)
            extracted_text = "\n".join([result[1] for result in results])
            return extracted_text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
