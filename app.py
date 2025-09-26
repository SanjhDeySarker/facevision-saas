from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import warnings
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from models.user import db, User  # Assumes models/user.py exists
import datetime
import json  # For metadata loading

# Suppress pkg_resources warnings (from previous fixes)
warnings.filterwarnings("ignore", module="pkg_resources")
warnings.filterwarnings("ignore", message=r".*pkg_resources.*deprecated.*")

# Import your models
from models.detect import detect_faces
from models.compare import compare_faces
from models.metadata import create_metadata, save_metadata

app = Flask(__name__)

# Database config (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# JWT config
app.config['JWT_SECRET_KEY'] = 'your-super-secret-key-change-in-prod'  # Change to a strong random string!
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=1)
jwt = JWTManager(app)

# Create tables (run once)
with app.app_context():
    db.create_all()

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
    """Simple health endpoint for testing API (public)."""
    return jsonify({"status": "healthy", "message": "Face SaaS API ready (local, free)"})

@app.route('/register', methods=['POST'])
def register():
    """Register new user (JSON: {"email": "user@example.com", "password": "pass123"})."""
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"success": False, "error": "Missing email or password"}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"success": False, "error": "Email already registered"}), 400
    try:
        user = User(email=data['email'])
        user.set_password(data['password'])
        db.session.add(user)
        db.session.commit()
        return jsonify({"success": True, "message": "User  registered successfully"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": f"Registration failed: {str(e)}"}), 500

@app.route('/login', methods=['POST'])
def login():
    """Login user (JSON: {"email": "user@example.com", "password": "pass123"}). Returns JWT token."""
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"success": False, "error": "Missing email or password"}), 400
    user = User.query.filter_by(email=data['email']).first()
    if user and user.check_password(data['password']) and user.is_active:
        access_token = create_access_token(identity=user.id)
        return jsonify({
            "success": True,
            "access_token": access_token,
            "user": user.to_dict()
        })
    return jsonify({"success": False, "error": "Invalid email or password"}), 401

@app.route('/detect', methods=['POST'])
@jwt_required()
def detect_route():
    """Detect faces in uploaded image (protected). Returns JSON with bounding boxes."""
    current_user_id = get_jwt_identity()
    print(f"Detection requested by user {current_user_id}")  # Optional logging
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
@jwt_required()
def compare_route():
    """Compare two uploaded images (protected). Returns JSON with match/distance."""
    current_user_id = get_jwt_identity()
    print(f"Comparison requested by user {current_user_id}")  # Optional logging
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
@jwt_required()
def metadata_route():
    """Create and save metadata (protected). Uses user_id from token."""
    current_user_id = get_jwt_identity()
    print(f"Metadata save requested by user {current_user_id}")  # Optional logging
    data = request.json
    if not data or 'image_path' not in data:
        return jsonify({"success": False, "error": "Missing image_path in JSON"}), 400
    try:
        # Always use authenticated user_id (secure; ignore client-provided)
        user_id = str(current_user_id)
        # Optional: Accept bbox or similarity_score from client
        bbox = data.get('bbox')
        similarity_score = data.get('similarity_score')
        if bbox:
            meta = create_metadata(user_id, data['image_path'], bbox=bbox)
        elif similarity_score:
            meta = create_metadata(user_id, data['image_path'], similarity_score=similarity_score)
        else:
            meta = create_metadata(user_id, data['image_path'])  # Basic
        save_result = save_metadata(meta)
        return jsonify(save_result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Metadata failed: {str(e)}"}), 500

@app.route('/metadata', methods=['GET'])
@jwt_required()
def get_metadata():
    """Retrieve user's saved metadata (protected, filtered by user)."""
    current_user_id = get_jwt_identity()
    print(f"Metadata retrieval requested by user {current_user_id}")  # Optional logging
    try:
        with open('face_metadata.json', 'r') as f:
            all_data = json.load(f)
        # Filter by current user only
        user_metadata = [item for item in all_data if item.get('user_id') == str(current_user_id)]
        return jsonify({"success": True, "metadata": user_metadata})
    except FileNotFoundError:
        return jsonify({"success": True, "metadata": []})
    except Exception as e:
        return jsonify({"success": False, "error": f"Retrieval failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)