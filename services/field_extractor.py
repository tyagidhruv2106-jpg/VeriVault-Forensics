import re

def extract_fields(text):
    fields = {
        "Name": "Not Detected",
        "DOB": "Not Detected",
        "Gender": "Not Detected",
        "Document Type": "Not Detected",
        "Document Number": "Not Detected"
    }
    
    if not text:
        return fields
        
    text_lower = text.lower()
    
    # Document Type Detection (Aadhaar prioritized)
    if "aadhaar" in text_lower or "government of india" in text_lower or "भारत सरकार" in text_lower or "dob" in text_lower:
        fields["Document Type"] = "Aadhaar"
        
    # DOB Detection
    dob_match = re.search(r'\b(\d{2}[/-]\d{2}[/-]\d{4}|\d{4})\b', text)
    if dob_match:
        fields["DOB"] = dob_match.group(0)
        
    # Gender Detection
    if "female" in text_lower:
        fields["Gender"] = "Female"
    elif "male" in text_lower:
        fields["Gender"] = "Male"
        
    # Aadhaar Number Detection (12 digits, can be spaced)
    aadhaar_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text)
    if aadhaar_match:
        num = aadhaar_match.group(0).replace(" ", "")
        if len(num) == 12:
            fields["Document Number"] = num
            
    # Name Heuristic (Just grabbing potential proper nouns before DOB line or keeping simple)
    # Since exact name extraction is hard without complex NER, we fallback to a simple heuristic
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 3]
    if lines:
        for i, line in enumerate(lines):
            if "dob" in line.lower() or "year of birth" in line.lower() or "yob" in line.lower():
                if i > 0 and not any(char.isdigit() for char in lines[i-1]):
                    # Simple heuristic: line above DOB is name
                    fields["Name"] = lines[i-1][:50]
                break
                
        # Fallback if name is still Not Detected
        if fields["Name"] == "Not Detected":
            for line in lines:
                lower_line = line.lower()
                # Skip lines with common keywords
                if not any(k in lower_line for k in ["aadhaar", "government", "सरकार", "dob", "year", "male", "female", "father"]):
                    # If it has no digits, assume it might be a name
                    if not any(c.isdigit() for c in line):
                        fields["Name"] = line[:50]
                        break

    return fields
