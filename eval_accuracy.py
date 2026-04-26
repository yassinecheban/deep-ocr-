import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
from ocr_model import (
    load_dataset_from_char_folders, build_tf_dataset,
    IMG_WIDTH, IMG_HEIGHT, decode_prediction, CHARACTERS
)

def evaluate_model():
    DATASET_PATH = "DATASET"
    MODEL_PATH = "checkpoints/ocr_predict.keras"
    BATCH_SIZE = 32

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return

    print("Loading data...")
    image_paths, labels = load_dataset_from_char_folders(DATASET_PATH)
    
    # Split for validation (same seed as in ocr_model.py)
    from sklearn.model_selection import train_test_split
    _, val_paths, _, val_labels = train_test_split(
        image_paths, labels, test_size=0.1, random_state=42
    )

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    val_ds = build_tf_dataset(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)

    num_batches = 20 # Test on ~640 images
    exact_matches = 0
    total_samples = 0
    
    print("Starting evaluation...")
    # val_ds yields (inputs, outputs) where inputs is a dict
    for batch in tqdm(val_ds.take(num_batches), total=num_batches):
        inputs, _ = batch
        images = inputs['image']
        labels_batch = inputs['label']
        
        preds = model.predict(images, verbose=0)
        
        for i in range(len(images)):
            total_samples += 1
            
            # Ground truth text
            # labels_batch[i] contains padded integer indices
            gt_indices = labels_batch[i].numpy()
            gt_text = "".join([CHARACTERS[idx-1] for idx in gt_indices if idx > 0])
            
            # Prediction text
            pred_text = decode_prediction(preds[i])
            
            if gt_text == pred_text:
                exact_matches += 1

    accuracy = (exact_matches / total_samples) * 100 if total_samples > 0 else 0
    print("\n--- Evaluation Results ---")
    print(f"Total Samples Tested: {total_samples}")
    print(f"Accuracy (Exact Match): {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
