import numpy as np
import face_recognition

def compare_two_encodings(enc1, enc2, threshold=0.6):
    """Compare two face encodings and return match result."""
    distance = float(np.linalg.norm(enc1 - enc2))
    return {"distance": distance, "match": distance <= threshold}
