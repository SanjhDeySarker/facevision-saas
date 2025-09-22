import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import os


def detect_faces(image_path):
    """
    Detect faces in an image using face_recognition (free, local).
    Ensures image is 8-bit RGB before detection.
    Returns dict with face count and bounding boxes.
    """
    temp_path = None
    try:
        # Step 1: Load and convert image to RGB with explicit format handling
        with Image.open(image_path) as pil_image:
            # Convert to RGB if not already
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Convert to numpy array directly
            image = np.array(pil_image)

            # Verify image format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Image must be RGB format")

        # Detect face locations
        face_locations = face_recognition.face_locations(image, model="hog")

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
                "confidence": 0.99  # placeholder since library doesn’t give confidence
            })

        return {
            "success": True,
            "faces_found": len(detected_faces),
            "faces": detected_faces,
            "message": "Face detection completed successfully (local, free)"
        }

    except FileNotFoundError:
        return {"success": False, "error": "Image file not found"}
    except ValueError as ve:
        return {"success": False, "error": f"Invalid image format: {str(ve)}"}
    except Exception as e:
        return {"success": False, "error": f"Detection failed: {str(e)}"}
    finally:
        # Always clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def visualize_detection(image_path, output_path="output_detection.jpg"):
    """
    Draw bounding boxes on the image and save it.
    """
    temp_path = None
    try:
        with Image.open(image_path) as pil_image:
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            base, _ = os.path.splitext(image_path)
            temp_path = f"{base}_temp_rgb.jpg"
            pil_image.save(temp_path, "JPEG", quality=95)

        image = face_recognition.load_image_file(temp_path)
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        face_locations = face_recognition.face_locations(image)
        for top, right, bottom, left in face_locations:
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)

        pil_image.save(output_path)
        print(f"Visualization saved to {output_path}")

    except Exception as e:
        print(f"Visualization failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    # Replace with your test image path
    result = detect_faces("test_image1.jpg")
    print("Detection Result:")
    print(result)

    # Optional visualization
    # visualize_detection("test_image1.jpg")
