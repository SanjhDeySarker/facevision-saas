import io
import base64
import cv2
import numpy as np
import face_recognition
from PIL import Image

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image bytes and return as NumPy array (RGB)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)

def detect_faces_in_image_bytes(image_bytes: bytes):
    """Detect faces, return bounding boxes + annotated image."""
    img = load_image_from_bytes(image_bytes)
    locations = face_recognition.face_locations(img, model="hog")

    faces = []
    for top, right, bottom, left in locations:
        faces.append({
            "box": {"top": int(top), "right": int(right),
                    "bottom": int(bottom), "left": int(left)}
        })

    annotated = _draw_boxes_and_b64(img, locations)
    return {"faces": faces, "count": len(faces), "annotated_image_b64": annotated}

def _draw_boxes_and_b64(img: np.ndarray, locations):
    """Draw rectangles on detected faces and return base64 image."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for (top, right, bottom, left) in locations:
        cv2.rectangle(bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    _, buf = cv2.imencode(".png", bgr)
    return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode()}"

def get_face_encodings_from_bytes(image_bytes: bytes):
    """Return 128-d embeddings for each detected face."""
    img = load_image_from_bytes(image_bytes)
    locations = face_recognition.face_locations(img, model="hog")
    return face_recognition.face_encodings(img, locations)
