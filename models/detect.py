import face_recognition
from PIL import Image, ImageDraw
import numpy as np

def detect_faces(image_path):
    """
    Detect faces in an image using face_recognition (free, local).
    Loads image, finds face locations, returns bounding boxes.
    Example usage: result = detect_faces('path/to/image.jpg')
    Returns: Dict with face count and list of bounding boxes (top, right, bottom, left).
    """
    try:
        # Load image using face_recognition (handles various formats)
        image = face_recognition.load_image_file(image_path)
        
        # Detect face locations (returns list of (top, right, bottom, left) tuples)
        face_locations = face_recognition.face_locations(image, model="hog")  # "hog" is fast and free; use "cnn" for accuracy (slower)
        
        # Prepare results
        detected_faces = []
        for top, right, bottom, left in face_locations:
            detected_faces.append({
                "bbox": {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left)
                },
                "confidence": 0.99  # face_recognition doesn't provide exact confidence; approximate high value
            })
        
        return {
            "success": True,
            "faces_found": len(detected_faces),
            "faces": detected_faces,
            "message": "Face detection completed successfully (local, free)"
        }
    
    except FileNotFoundError:
        return {"success": False, "error": "Image file not found"}
    except Exception as e:
        return {"success": False, "error": f"Detection failed: {str(e)}"}

# Optional: Visualize results (saves an image with boxes drawn)
def visualize_detection(image_path, output_path="output_detection.jpg"):
    """
    Draws bounding boxes on the image and saves it (for testing/debugging).
    """
    image = face_recognition.load_image_file(image_path)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    face_locations = face_recognition.face_locations(image)
    for top, right, bottom, left in face_locations:
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
    
    pil_image.save(output_path)
    print(f"Visualization saved to {output_path}")

# Test the function (run this file directly)
if __name__ == "__main__":
    # Replace with your test image
    result = detect_faces("test_image1.jpg")
    print("Detection Result:")
    print(result)
    
    # Optional: Visualize
    # visualize_detection("test_image1.jpg")