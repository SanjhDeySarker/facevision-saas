import json
import datetime
import os
from typing import Dict, List

METADATA_FILE = "face_metadata.json"  # Free JSON file in project root

def create_metadata(user_id: str, image_path: str, bbox: Dict = None, similarity_score: float = None) -> Dict:
    """
    Create metadata for a face (e.g., after detection/comparison).
    Includes user ID, timestamp, bbox, and optional score.
    Returns: Dict ready for storage.
    """
    timestamp = datetime.datetime.now().isoformat()
    
    metadata = {
        "user_id": user_id,
        "image_path": image_path,
        "timestamp": timestamp,
        "bbox": bbox or {},  # From detect.py
        "similarity_score": similarity_score,  # From compare.py
        "status": "processed"
    }
    return metadata

def save_metadata(metadata: Dict, append: bool = True):
    """
    Save metadata to JSON file (free, local storage).
    If append=True, adds to existing list; else, overwrites.
    """
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        else:
            data = []
        
        if append:
            data.append(metadata)
        else:
            data = [metadata]  # Overwrite with single entry
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        
        return {"success": True, "message": f"Metadata saved to {METADATA_FILE}"}
    
    except Exception as e:
        return {"success": False, "error": f"Save failed: {str(e)}"}

def load_metadata(user_id: str = None) -> List[Dict]:
    """
    Load all or user-specific metadata from JSON.
    Returns: List of metadata dicts.
    """
    try:
        if not os.path.exists(METADATA_FILE):
            return []
        
        with open(METADATA_FILE, 'r') as f:
            data = json.load(f)
        
        if user_id:
            return [item for item in data if item.get("user_id") == user_id]
        return data
    
    except Exception as e:
        return [{"error": f"Load failed: {str(e)}"}]

# Test the functions
if __name__ == "__main__":
    # Example: Create metadata from detection/comparison
    sample_bbox = {"top": 150, "right": 400, "bottom": 300, "left": 200}
    sample_metadata = create_metadata("user123", "test_image1.jpg", sample_bbox, 0.45)
    print("Created Metadata:")
    print(sample_metadata)
    
    # Save it
    save_result = save_metadata(sample_metadata)
    print("Save Result:", save_result)
    
    # Load it
    loaded = load_metadata("user123")
    print("Loaded Metadata:")
    print(loaded)