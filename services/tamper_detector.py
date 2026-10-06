import cv2
import numpy as np
import os

def detect_tamper(image_path):
    """
    Performs a basic Error Level Analysis (ELA) demonstration.
    Returns: 'CLEAN', 'POTENTIALLY SUSPICIOUS', or 'ERROR'
    """
    temp_path = "temp_ela.jpg"
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "ERROR", "Unable to read image."
            
        # Save a compressed version
        cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_compressed = cv2.imread(temp_path)
        
        # Absolute difference between original and compressed
        diff = cv2.absdiff(img, img_compressed)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Basic heuristic thresholds for demo
        if max_diff > 60 or mean_diff > 2.0:
            return "POTENTIALLY SUSPICIOUS", f"Unusual compression differences detected (Max: {max_diff}, Mean: {mean_diff:.2f}). Manual review recommended."
        else:
            return "CLEAN", "No significant suspicious compression region detected."
            
    except Exception as e:
        print(f"Tamper detection error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return "ERROR", "Tamper analysis failed."
