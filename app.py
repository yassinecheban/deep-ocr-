import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
from ocr_model import CHARACTERS, IMG_WIDTH, IMG_HEIGHT, decode_prediction

app = Flask(__name__)
# Allow requests from any origin (for development)
# In production, replace '*' with your frontend domain
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
IMAGE_WIDTH = IMG_WIDTH
IMAGE_HEIGHT = IMG_HEIGHT
MODEL_PATH = "checkpoints/ocr_predict.keras"

# Mappings
int_to_char = {i + 1: ch for i, ch in enumerate(CHARACTERS)}

# Load Model
# The checkpoints are HDF5 weights-only files (saved with older Keras).
# We rebuild the prediction model architecture and load weights manually.
FALLBACK_MODEL_PATH = "checkpoints/ocr_best.keras"

def load_weights_from_hdf5(model, filepath):
    """Manually load weights from a Keras 2-style HDF5 weights file."""
    import h5py
    with h5py.File(filepath, 'r') as f:
        weight_group = f['model_weights']
        layer_names = [
            n.decode('utf8') if isinstance(n, bytes) else n
            for n in weight_group.attrs.get('layer_names', [])
        ]
        for layer in model.layers:
            if layer.name in layer_names:
                g = weight_group[layer.name]
                weight_names = [
                    n.decode('utf8') if isinstance(n, bytes) else n
                    for n in g.attrs.get('weight_names', [])
                ]
                weights = [g[wn][()] for wn in weight_names]
                if weights:
                    layer.set_weights(weights)

def load_prediction_model(weights_path):
    """
    Rebuild the prediction model from ocr_model.py and load saved weights.
    Handles both modern .keras zip format and legacy HDF5 weights-only files.
    """
    import zipfile, h5py
    from ocr_model import build_ocr_model, VOCAB_SIZE

    # Check if it's a full model save (zip) or weights-only (HDF5)
    is_zip = False
    try:
        with zipfile.ZipFile(weights_path, 'r'):
            is_zip = True
    except zipfile.BadZipFile:
        pass

    if is_zip:
        # Modern full-model .keras format
        return tf.keras.models.load_model(weights_path, compile=False)

    # HDF5 file — check structure
    with h5py.File(weights_path, 'r') as f:
        top_keys = list(f.keys())

    if 'model_weights' in top_keys:
        # Weights-only save: rebuild architecture and load weights manually
        print("  (HDF5 weights-only file — rebuilding model and loading weights)")
        _, prediction_model = build_ocr_model(IMG_WIDTH, IMG_HEIGHT, VOCAB_SIZE)
        load_weights_from_hdf5(prediction_model, weights_path)
        return prediction_model
    else:
        # Full HDF5 model save (older Keras 2 format)
        print("  (HDF5 full-model file)")
        return tf.keras.models.load_model(weights_path, compile=False)


if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = load_prediction_model(MODEL_PATH)
        print(f"Model loaded successfully. Output shape: {model.output_shape}")
    except Exception as e:
        print(f"WARNING: Failed to load {MODEL_PATH}: {e}")
        if os.path.exists(FALLBACK_MODEL_PATH):
            print(f"Trying fallback: {FALLBACK_MODEL_PATH}...")
            try:
                model = load_prediction_model(FALLBACK_MODEL_PATH)
                print(f"Fallback model loaded. Output shape: {model.output_shape}")
            except Exception as e2:
                print(f"WARNING: Fallback also failed: {e2}")
                model = None
        else:
            model = None
else:
    print(f"WARNING: Model not found at {MODEL_PATH}. Prediction will fail until model is trained.")
    model = None

def preprocess_single_char(img):
    """
    Match the preprocessing logic in ocr_model.py.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu binarization
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize: keep aspect ratio, pad with white
    h, w = img.shape
    ratio = IMAGE_WIDTH / w
    new_w = IMAGE_WIDTH
    new_h = int(h * ratio)

    if new_h > IMAGE_HEIGHT:
        ratio = IMAGE_HEIGHT / h
        new_h = IMAGE_HEIGHT
        new_w = int(w * ratio)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to target size
    pad_top    = (IMAGE_HEIGHT - new_h) // 2
    pad_bottom = IMAGE_HEIGHT - new_h - pad_top
    pad_left   = (IMAGE_WIDTH  - new_w) // 2
    pad_right  = IMAGE_WIDTH  - new_w - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=255)

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.transpose(img, (1, 0, 2))  # (W, H, 1)
    img = np.expand_dims(img, axis=0)   # (1, W, H, 1)
    return img

def run_prediction(img):
    if model is None:
        return "Model not loaded"
    pred = model.predict(img, verbose=0)
    result = decode_prediction(pred[0])
    # Strip spurious leading 'z' — a known artifact of the current model checkpoint
    if result.startswith('z') and len(result) > 1:
        result = result[1:]
    return result

def segment_and_predict(image_bytes):
    # 1. Decode and Binarize
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Otsu thresholding
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Find contours (characters)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Filter and Sort bounding boxes
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2: # Filter tiny noise
            rects.append((x, y, w, h))
            
    print(f"--- New Prediction Request ---")
    print(f"Found {len(rects)} potential characters.")
            
    if not rects:
        return ""

    # Sort by lines (group by y with some tolerance)
    rects.sort(key=lambda r: r[1])
    lines = []
    if rects:
        curr_line = [rects[0]]
        for i in range(1, len(rects)):
            if rects[i][1] < curr_line[-1][1] + curr_line[-1][3] * 0.7:
                curr_line.append(rects[i])
            else:
                lines.append(sorted(curr_line, key=lambda r: r[0]))
                curr_line = [rects[i]]
        lines.append(sorted(curr_line, key=lambda r: r[0]))

    # 4. Predict each character
    final_text = ""
    for i, line in enumerate(lines):
        line_text = ""
        for j, (x, y, w, h) in enumerate(line):
            # Crop with significant padding for better recognition
            pad = int(max(w, h) * 0.2)
            y1 = max(0, y - pad)
            y2 = min(img_gray.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(img_gray.shape[1], x + w + pad)
            
            char_img = img_gray[y1:y2, x1:x2]
            
            # Preprocess and predict
            proc_img = preprocess_single_char(char_img)
            char = run_prediction(proc_img)
            print(f"  Line {i}, Char {j}: Raw bbox=[{x},{y},{w},{h}], Predicted='{char}'")
            line_text += char
        final_text += line_text + " " # Space between words if multiple
        
    print(f"Final Prediction: '{final_text.strip()}'")
    return final_text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api')
def api_info():
    return jsonify({
        "service": "OCR API",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Upload image for OCR prediction"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    try:
        # Use segmentation to handle multiple characters
        result_text = segment_and_predict(img_bytes)
        
        return jsonify({
            "prediction": result_text,
            "status": "success"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
