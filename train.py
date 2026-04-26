import os
import tensorflow as tf
from ocr_model import train

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Successfully configured GPU memory growth.")
        except RuntimeError as e:
            print(f"GPU Configuration Error: {e}")

def main():
    # Configuration
    DATASET_PATH = "DATASET"
    CHECKPOINT_DIR = "checkpoints"
    
    configure_gpu()

    print(f"Starting OCR Training using {DATASET_PATH}...")
    
    # Run the training pipeline from ocr_model.py
    # Setting use_char_folders=True to support your existing folder-per-character structure
    train(
        dataset_dir=DATASET_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
        use_char_folders=True
    )

if __name__ == "__main__":
    main()
