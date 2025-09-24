from models import detect, compare, metadata

def run_tests():
    # Load sample images
    with open("tests/sample1.jpg", "rb") as f1, open("tests/sample2.jpg", "rb") as f2:
        img1 = f1.read()
        img2 = f2.read()

    # Face detection
    result = detect.detect_faces_in_image_bytes(img1)
    print("Faces detected:", result["count"])

    # Face encodings + comparison
    enc1 = detect.get_face_encodings_from_bytes(img1)[0]
    enc2 = detect.get_face_encodings_from_bytes(img2)[0]
    comp = compare.compare_two_encodings(enc1, enc2)
    print("Comparison:", comp)

    # Metadata
    meta = metadata.analyze_image_bytes(img1)
    print("Metadata:", meta)

if __name__ == "__main__":
    run_tests()
