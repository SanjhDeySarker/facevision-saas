from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import warnings

# Suppress pkg_resources warnings (from previous fixes)
warnings.filterwarnings("ignore", module="pkg_resources")
warnings.filterwarnings("ignore", message=r".*pkg_resources.*deprecated.*")

# Import your models
from models.detect import detect_faces
from models.compare import compare_faces
from models.metadata import create_metadata, save_metadata

app = Flask(__name__)

# Config: Temp upload folder (creates if missing)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}  # Supported image types

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health endpoint for testing API."""
    return jsonify({"status": "healthy", "message": "Face SaaS API ready (local, free)"})

@app.route('/detect', methods=['POST'])
def detect_route():
    """Detect faces in uploaded image. Returns JSON with bounding boxes."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            result = detect_faces(filepath)
            # Clean up uploaded file after processing
            os.remove(filepath)
            return jsonify(result)
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"success": False, "error": f"Detection failed: {str(e)}"}), 500
    return jsonify({"success": False, "error": "Unsupported file type"}), 400

@app.route('/compare', methods=['POST'])
def compare_route():
    """Compare two uploaded images. Returns JSON with match/distance."""
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"success": False, "error": "Missing file1 or file2"}), 400
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' or file2.filename == '':
        return jsonify({"success": False, "error": "No files selected"}), 400
    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(filepath1)
        file2.save(filepath2)
        try:
            result = compare_faces(filepath1, filepath2, tolerance=0.6)
            # Clean up
            for fp in [filepath1, filepath2]:
                if os.path.exists(fp):
                    os.remove(fp)
            return jsonify(result)
        except Exception as e:
            # Clean up on error
            for fp in [filepath1, filepath2]:
                if os.path.exists(fp):
                    os.remove(fp)
            return jsonify({"success": False, "error": f"Comparison failed: {str(e)}"}), 500
    return jsonify({"success": False, "error": "Unsupported file types"}), 400

@app.route('/metadata', methods=['POST'])
def metadata_route():
    """Create and save metadata from request data (e.g., after detection/compare)."""
    data = request.json
    if not data or 'user_id' not in data or 'image_path' not in data:
        return jsonify({"success": False, "error": "Missing user_id or image_path in JSON"}), 400
    try:
        # Optional: Accept bbox or similarity_score from client
        bbox = data.get('bbox')
        similarity_score = data.get('similarity_score')
        if bbox:
            meta = create_metadata(data['user_id'], data['image_path'], bbox=bbox)
        elif similarity_score:
            meta = create_metadata(data['user_id'], data['image_path'], similarity_score=similarity_score)
        else:
            meta = create_metadata(data['user_id'], data['image_path'])  # Basic
        save_result = save_metadata(meta)
        return jsonify(save_result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Metadata failed: {str(e)}"}), 500

@app.route('/metadata', methods=['GET'])
def get_metadata():
    """Retrieve all saved metadata (for demo)."""
    try:
        with open('face_metadata.json', 'r') as f:
            import json
            data = json.load(f)
        return jsonify({"success": True, "metadata": data})
    except FileNotFoundError:
        return jsonify({"success": True, "metadata": []})
    except Exception as e:
        return jsonify({"success": False, "error": f"Retrieval failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)