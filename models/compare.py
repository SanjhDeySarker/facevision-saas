import face_recognition
import numpy as np

def compare_faces(image1_path, image2_path, tolerance=0.6):
    """
    Compare two faces using face_recognition (free, local).
    Loads images, generates encodings, computes distance, returns similarity score.
    Example: result = compare_faces('img1.jpg', 'img2.jpg')
    Returns: Dict with match verdict and distance (0 = identical, <0.6 = match).
    tolerance: Adjust for strictness (default 0.6 for ~99% accuracy).
    """
    try:
        # Load images and get encodings (128D vectors)
        image1 = face_recognition.load_image_file(image1_path)
        image2 = face_recognition.load_image_file(image2_path)
        
        # Get face encodings (assumes one main face per image; extend for multi-face)
        encoding1 = face_recognition.face_encodings(image1)[0]  # First face
        encoding2 = face_recognition.face_encodings(image2)[0]  # First face
        
        # Compute distance (Euclidean)
        distance = face_recognition.face_distance([encoding1], encoding2)[0]
        
        # Determine match
        is_match = distance < tolerance
        
        return {
            "success": True,
            "match": is_match,
            "distance": float(distance),  # Similarity score (lower = better match)
            "tolerance": tolerance,
            "message": f"Comparison completed (local, free). Match: {is_match}"
        }
    
    except IndexError:
        return {"success": False, "error": "No faces found in one or both images"}
    except FileNotFoundError:
        return {"success": False, "error": "One or both image files not found"}
    except Exception as e:
        return {"success": False, "error": f"Comparison failed: {str(e)}"}

# Test the function (run this file directly)
if __name__ == "__main__":
    # Replace with your test images (use same person for match, different for no-match)
    result = compare_faces("test_image1.jpg", "test_image2.jpg")
    print("Comparison Result:")
    print(result)