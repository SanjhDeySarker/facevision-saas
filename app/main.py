# app/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import JSONResponse
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from app.utils import load_image_from_bytes, save_temp_file_from_bytes, load_image_from_url
from numpy.linalg import norm
import cv2, io

app = FastAPI(title="Free Face AI SaaS (MVP)")

# --- mediapipe setup (singleton)
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

@app.post("/api/v1/face/detect")
async def detect(file: UploadFile = File(None), url: str = Form(None)):
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide file or url")
    if file:
        raw = await file.read()
    else:
        raw = load_image_from_url(url)
    img_np = load_image_from_bytes(raw)
    # MediaPipe expects RGB images as numpy arrays
    results = mp_face.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    faces = []
    if results.detections:
        h, w, _ = img_np.shape
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            faces.append({
                "bbox": [
                    int(box.xmin * w),
                    int(box.ymin * h),
                    int((box.xmin + box.width) * w),
                    int((box.ymin + box.height) * h),
                ],
                "score": float(det.score[0]) if det.score else None
            })
    return {"faces": faces}

@app.post("/api/v1/face/metadata")
async def metadata(file: UploadFile = File(None), url: str = Form(None)):
    if file:
        raw = await file.read()
    elif url:
        raw = load_image_from_url(url)
    else:
        raise HTTPException(status_code=400, detail="Provide file or url")
    tmp_path = save_temp_file_from_bytes(raw)
    try:
        # DeepFace analyze returns dict with age/gender/emotion keys
        analysis = DeepFace.analyze(img_path=tmp_path, actions=['age','gender','emotion'], enforce_detection=False)
        return JSONResponse(content=analysis)
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.post("/api/v1/face/compare")
async def compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    b1 = await file1.read(); p1 = save_temp_file_from_bytes(b1)
    b2 = await file2.read(); p2 = save_temp_file_from_bytes(b2)
    try:
        rep1 = DeepFace.represent(img_path=p1, enforce_detection=False)
        rep2 = DeepFace.represent(img_path=p2, enforce_detection=False)
        score = cosine_sim(rep1, rep2)
        match = score > 0.5  # threshold you can tune
        return {"score": score, "match": bool(match)}
    finally:
        import os
        os.remove(p1); os.remove(p2)
