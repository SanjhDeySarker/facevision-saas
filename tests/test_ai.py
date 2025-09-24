from models import detect, compare, metadata

def run_tests():
    # Load sample images
    with open("tests/sample1.jpg", "rb") as f1, open("tests/sample2.jpg", "rb") as f2:
        img1_bytes = f1.read()
        img2_bytes = f2.read()

    # Debug: image info
    img1_np = detect.load_image_from_bytes(img1_bytes)
    img2_np = detect.load_image_from_bytes(img2_bytes)
    print("Sample1 Image shape:", img1_np.shape, "dtype:", img1_np.dtype)
    print("Sample2 Image shape:", img2_np.shape, "dtype:", img2_np.dtype)

    # Face Detection
    result1 = detect.detect_faces_in_image_bytes(img1_bytes)
    print("\nFaces detected in sample1:", result1["count"])
    if result1["count"] > 0:
        print("Annotated image (base64, first 100 chars):", result1["annotated_image_b64"][:100])

    # Face Encodings + Comparison
    enc1_list = detect.get_face_encodings_from_bytes(img1_bytes)
    enc2_list = detect.get_face_encodings_from_bytes(img2_bytes)

    if len(enc1_list) == 0 or len(enc2_list) == 0:
        print("\nNo faces found in one of the images. Cannot compare.")
    else:
        enc1 = enc1_list[0]
        enc2 = enc2_list[0]
        comparison = compare.compare_two_encodings(enc1, enc2)
        print("\nFace Comparison Result:", comparison)

    # Metadata Extraction
    meta1 = metadata.analyze_image_bytes(img1_bytes)
    print("\nMetadata for sample1:", meta1)


if __name__ == "__main__":
    run_tests()
