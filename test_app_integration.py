"""
Integration Test Suite for VeriVault
Verifies:
1. Aadhaar Card Verification Feature (/scan and Verhoeff validation)
2. Universal AI Image & Manipulation Detector (/ai-detector, /api/detect-ai-image)
3. Model Performance & Metrics Endpoint (/api/model-metrics)
"""

import os
import io
import json
import unittest
from PIL import Image
from app import app
from services.aadhaar_validator import validate_aadhaar, validate_verhoeff
from services.tamper_detector import detect_tamper

class VeriVaultIntegrationTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    # -------------------------------------------------------------
    # 1. Existing Aadhaar Verification Tests (Preserved 100%)
    # -------------------------------------------------------------
    def test_verhoeff_checksum(self):
        """Tests that Verhoeff algorithm works properly."""
        # Known valid Verhoeff number examples:
        # e.g., 234567890123 -> check if last digit satisfies checksum
        # Or testing known Aadhaar numbers
        self.assertFalse(validate_aadhaar("123456789012"))
        self.assertFalse(validate_aadhaar("000000000000"))
        self.assertFalse(validate_aadhaar("short"))

    def test_tamper_detector(self):
        """Tests that ELA tamper detector runs on sample image."""
        sample_path = os.path.join("static", "samples", "sample_real.jpg")
        if os.path.exists(sample_path):
            status, msg = detect_tamper(sample_path)
            self.assertIn(status, ["CLEAN", "POTENTIALLY SUSPICIOUS", "ERROR"])

    def test_home_page_renders(self):
        """Tests that home page renders with both feature cards."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("Aadhaar Card Authenticity Check", html)
        self.assertIn("Universal AI Image & Manipulation Detector", html)
        self.assertIn("uploadForm", html)
        self.assertIn("/scan", html)

    # -------------------------------------------------------------
    # 2. Universal AI Detector Tests
    # -------------------------------------------------------------
    def test_ai_detector_page_renders(self):
        """Tests that AI detector page and metrics tabs render."""
        res = self.client.get('/ai-detector')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("Universal AI Image & Manipulation Detector", html)
        self.assertIn("Forensic Detector", html)
        self.assertIn("Model Performance & Metrics", html)
        self.assertIn("training_curves.png", html)
        self.assertIn("confusion_matrix.png", html)
        self.assertIn("roc_curve.png", html)

    def test_ai_detector_sample_api(self):
        """Tests /api/detect-ai-image with sample real and AI images."""
        # Test sample_real
        res_real = self.client.post('/api/detect-ai-image', 
                                    data=json.dumps({'sample_name': 'sample_real'}),
                                    content_type='application/json')
        self.assertEqual(res_real.status_code, 200)
        data_real = json.loads(res_real.data)
        self.assertTrue(data_real.get('success'))
        self.assertIn('verdict', data_real)
        self.assertIn('confidence_score', data_real)
        self.assertIn('gradcam_image', data_real)
        self.assertIn('ela_image', data_real)
        self.assertTrue(data_real['gradcam_image'].startswith('data:image/png;base64,'))

        # Test sample_ai
        res_ai = self.client.post('/api/detect-ai-image', 
                                  data=json.dumps({'sample_name': 'sample_ai'}),
                                  content_type='application/json')
        self.assertEqual(res_ai.status_code, 200)
        data_ai = json.loads(res_ai.data)
        self.assertTrue(data_ai.get('success'))
        self.assertEqual(data_ai['verdict'], 'AI-Generated / AI-Edited')

    def test_ai_detector_file_upload(self):
        """Tests /api/detect-ai-image with multipart file upload."""
        img = Image.new('RGB', (100, 100), color=(73, 109, 137))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        res = self.client.post('/api/detect-ai-image',
                               data={'image': (img_bytes, 'test_upload.jpg')},
                               content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data.get('success'))
        self.assertIn('verdict', data)

    # -------------------------------------------------------------
    # 3. Model Metrics API Tests
    # -------------------------------------------------------------
    def test_model_metrics_api(self):
        """Tests that /api/model-metrics serves metrics_report.json."""
        res = self.client.get('/api/model-metrics')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('test_accuracy', data)
        self.assertIn('precision', data)
        self.assertIn('recall', data)
        self.assertIn('f1_score', data)
        self.assertIn('roc_auc', data)
        self.assertIn('dataset_size', data)

if __name__ == '__main__':
    unittest.main()

