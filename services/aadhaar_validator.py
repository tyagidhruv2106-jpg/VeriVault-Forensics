# Verhoeff algorithm logic for Aadhaar validation
verhoeff_d = [
    [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5], [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7], [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3], [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0]
]
verhoeff_p = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4], [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7], [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8]
]

def validate_verhoeff(num):
    c = 0
    for i, p in enumerate(reversed(str(num))):
        c = verhoeff_d[c][verhoeff_p[i % 8][int(p)]]
    return c == 0

def validate_aadhaar(aadhaar_number):
    """
    Validates a 12-digit Aadhaar number using the Verhoeff checksum.
    Returns: True if valid, False otherwise.
    """
    if not aadhaar_number or len(str(aadhaar_number)) != 12:
        return False
    return validate_verhoeff(aadhaar_number)
