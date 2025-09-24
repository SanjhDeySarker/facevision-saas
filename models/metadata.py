from deepface import DeepFace
import numpy as np
from PIL import Image
import io

def analyze_image_bytes(image_bytes: bytes, actions=None):
    """Extract metadata like age, gender, emotion."""
    if actions is None:
        actions = ["age", "gender", "emotion"]

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_arr = np.array(img)

    try:
        result = DeepFace.analyze(img_arr, actions=actions, enforce_detection=True)
        return result if isinstance(result, dict) else {"faces": result}
    except Exception as e:
        return {"error": str(e)}
