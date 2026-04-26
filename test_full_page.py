import os
import numpy as np
import tensorflow as tf
import cv2
from ocr_model import (
    IMG_WIDTH, IMG_HEIGHT, decode_prediction, CHARACTERS, preprocess_image
)

# Configuration
MODEL_PATH = "checkpoints/ocr_predict.keras"

# Load Model
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
else:
    print(f"Error: Model not found at {MODEL_PATH}")
    model = None

def preprocess_single_char(img_gray):
    """
    Standardize character image for the model.
    """
    # Otsu binarization (inverted because segmentation usually finds black on white)
    # The training code expects white background (255) and black text.
    # If the input is already black text on white, we just need to ensure background is 255.
    
    # Resize keeping aspect ratio and padding with white (255)
    h, w = img_gray.shape
    ratio = IMG_WIDTH / w
    new_w = IMG_WIDTH
    new_h = int(h * ratio)

    if new_h > IMG_HEIGHT:
        ratio = IMG_HEIGHT / h
        new_h = IMG_HEIGHT
        new_w = int(w * ratio)

    img = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to target size with 255 (white)
    pad_top    = (IMG_HEIGHT - new_h) // 2
    pad_bottom = IMG_HEIGHT - new_h - pad_top
    pad_left   = (IMG_WIDTH  - new_w) // 2
    pad_right  = IMG_WIDTH  - new_w - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=255)

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.transpose(img, (1, 0, 2))  # (W, H, 1)
    img = np.expand_dims(img, axis=0)
    return img

def test_full_page(img_path):
    if model is None: return

    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Error: Could not read {img_path}")
        return

    # Otsu thresholding (invert to find contours)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 3 and h > 3: # Noise filter
            rects.append((x, y, w, h))
            
    if not rects:
        print("No characters found!")
        return

    # Sort by lines (y coordinate with some tolerance)
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

    print(f"\n--- Recognizing: {img_path} ---")
    
    for i, line in enumerate(lines):
        line_text = ""
        for (x, y, w, h) in line:
            # Crop with padding
            pad = int(max(w, h) * 0.2)
            y1, y2 = max(0, y-pad), min(img_gray.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(img_gray.shape[1], x+w+pad)
            char_img = img_gray[y1:y2, x1:x2]
            
            # Predict
            proc_img = preprocess_single_char(char_img)
            pred = model.predict(proc_img, verbose=0)
            char = decode_prediction(pred[0])
            line_text += char
        print(f"Line {i}: {line_text}")
    
    print("\n--- End of Recognition ---")

if __name__ == "__main__":
    # Test on a sample image if it exists
    sample = "Untitled.png"
    if os.path.exists(sample):
        test_full_page(sample)
    else:
        print(f"Sample image {sample} not found.")
