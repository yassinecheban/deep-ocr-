"""
Full OCR Training Pipeline for Printed/Typed Text
CRNN + CTC Loss — TensorFlow/Keras
"""

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
)
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

IMG_WIDTH    = 128   # Must be divisible by 8 (3 MaxPool2D layers)
IMG_HEIGHT   = 32    # Must be divisible by 8
BATCH_SIZE   = 32
EPOCHS       = 100
LEARNING_RATE = 1e-3

# Characters your model will recognize
# Adjust this to match your dataset's character set
CHARACTERS = sorted(list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-()/"
))
VOCAB_SIZE   = len(CHARACTERS)

# Mappings
char_to_int = {ch: i + 1 for i, ch in enumerate(CHARACTERS)}  # 0 reserved for CTC blank
int_to_char = {i + 1: ch for i, ch in enumerate(CHARACTERS)}

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Characters: {''.join(CHARACTERS)}")


# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────
# Expected folder structure:
#   dataset/
#     images/
#       word_001.png
#       word_002.png
#     labels.txt          <- "word_001.png hello world"  (one per line)
#
# OR use IAM / MJSynth datasets directly — see load_mjsynth() below.

def load_dataset_from_folder(dataset_dir: str):
    """
    Load images + labels from a simple flat folder with a labels.txt file.
    Each line in labels.txt: <filename> <label text>
    """
    dataset_dir = Path(dataset_dir)
    labels_file = dataset_dir / "labels.txt"
    images_dir  = dataset_dir / "images"

    image_paths, labels = [], []

    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            fname, label = parts
            img_path = images_dir / fname
            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(label)

    print(f"Loaded {len(image_paths)} samples from {dataset_dir}")
    return image_paths, labels


def load_mjsynth(dataset_dir: str, max_samples: int = 100_000):
    """
    Loader for MJSynth (Synth90k) dataset.
    Download: https://www.robots.ox.ac.uk/~vgg/data/text/
    Folder structure: mnt/<wordclass>/<lexicon>/<img>.jpg
    annotation_train.txt format: ./path/to/img.jpg <label_index>
    lexicon.txt: one word per line
    """
    dataset_dir = Path(dataset_dir)
    lexicon_path = dataset_dir / "lexicon.txt"
    annot_path   = dataset_dir / "annotation_train.txt"

    with open(lexicon_path) as f:
        lexicon = [line.strip() for line in f]

    image_paths, labels = [], []
    with open(annot_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            parts = line.strip().split(" ")
            rel_path, idx = parts[0], int(parts[1])
            full_path = dataset_dir / rel_path.lstrip("./")
            if full_path.exists():
                image_paths.append(str(full_path))
                labels.append(lexicon[idx])

    print(f"Loaded {len(image_paths)} MJSynth samples")
    return image_paths, labels


def load_dataset_from_char_folders(dataset_dir: str):
    """
    Load images + labels from a folder structure where each subfolder name
    represents a character (e.g., 'A_U' -> 'A', 'a_L' -> 'a', '0' -> '0').
    """
    dataset_dir = Path(dataset_dir)
    image_paths, labels = [], []

    # Map folder names to characters
    folder_map = {
        "comma": ",", "dash": "-", "dot": ".", "slash": "/"
    }
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        folder_map[f"{c}_U"] = c
    for c in "abcdefghijklmnopqrstuvwxyz":
        folder_map[f"{c}_L"] = c
    for i in range(10):
        folder_map[str(i)] = str(i)

    print(f"Scanning directory: {dataset_dir}")
    for folder in dataset_dir.iterdir():
        if not folder.is_dir():
            continue

        char = folder_map.get(folder.name)
        if char is None:
            continue

        for img_path in folder.glob("*"):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                image_paths.append(str(img_path))
                labels.append(char)

    print(f"Loaded {len(image_paths)} samples from {dataset_dir}")
    return image_paths, labels


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_image(img_path: str,
                     img_width: int = IMG_WIDTH,
                     img_height: int = IMG_HEIGHT) -> np.ndarray:
    """
    Load, grayscale, binarize, resize (keeping aspect ratio),
    normalize to [0, 1], shape: (img_width, img_height, 1).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Otsu binarization — great for printed text
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize: keep aspect ratio, pad with white
    h, w = img.shape
    ratio = img_width / w
    new_w = img_width
    new_h = int(h * ratio)

    if new_h > img_height:
        ratio = img_height / h
        new_h = img_height
        new_w = int(w * ratio)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to target size
    pad_top    = (img_height - new_h) // 2
    pad_bottom = img_height - new_h - pad_top
    pad_left   = (img_width  - new_w) // 2
    pad_right  = img_width  - new_w - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=255)

    img = img.astype(np.float32) / 255.0          # normalize
    img = np.expand_dims(img, axis=-1)             # (H, W, 1)
    img = np.transpose(img, (1, 0, 2))             # (W, H, 1) = (img_width, img_height, 1)
    return img


def encode_label(label: str) -> list[int]:
    """Convert text string to list of integer indices."""
    return [char_to_int[ch] for ch in label if ch in char_to_int]


def decode_prediction(pred: np.ndarray) -> str:
    """
    Greedy CTC decode: argmax -> remove blanks -> remove duplicates.
    pred shape: (time_steps, vocab_size + 1)
    """
    indices = np.argmax(pred, axis=-1)
    decoded = []
    prev = None
    for idx in indices:
        if idx != prev and idx != 0:   # 0 = CTC blank
            decoded.append(int_to_char.get(idx, ""))
        prev = idx
    return "".join(decoded)


# ─────────────────────────────────────────────
# 4. tf.data PIPELINE
# ─────────────────────────────────────────────

def build_tf_dataset(image_paths: list,
                     labels: list,
                     batch_size: int = BATCH_SIZE,
                     augment: bool = False,
                     shuffle: bool = True):
    """
    Build a tf.data.Dataset that yields:
        {"image": ..., "label": ..., "input_length": ..., "label_length": ...}
    with CTC loss output shape (batch,).
    """

    def load_sample(img_path, label_str):
        # Decode tensors
        img_path  = img_path.numpy().decode("utf-8")
        label_str = label_str.numpy().decode("utf-8")

        img     = preprocess_image(img_path)
        encoded = encode_label(label_str)

        label_len   = len(encoded)
        # CTC input length = time steps after CNN = img_width // 8
        input_len   = IMG_WIDTH // 8

        return img, encoded, input_len, label_len

    def tf_load_sample(img_path, label_str):
        img, encoded, input_len, label_len = tf.py_function(
            load_sample,
            [img_path, label_str],
            [tf.float32, tf.int64, tf.int64, tf.int64]
        )
        img.set_shape([IMG_WIDTH, IMG_HEIGHT, 1])
        return img, encoded, input_len, label_len

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 5000))

    dataset = dataset.map(tf_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    # Pad variable-length labels within each batch
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [IMG_WIDTH, IMG_HEIGHT, 1],   # image
            [None],                        # label
            [],                            # input_length
            [],                            # label_length
        ),
        padding_values=(
            0.0, 
            tf.cast(0, tf.int64), 
            tf.cast(0, tf.int64), 
            tf.cast(0, tf.int64)
        )
    )

    # Reformat into model's expected input dict
    def reformat(img, label, input_len, label_len):
        inputs = {
            "image":        img,
            "label":        label,
            "input_length": tf.expand_dims(input_len, -1),
            "label_length": tf.expand_dims(label_len, -1),
        }
        # CTC loss layer outputs a single value per sample; use zeros as dummy target
        outputs = tf.zeros([tf.shape(img)[0]])
        return inputs, outputs

    dataset = dataset.map(reformat, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ─────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────

def build_ocr_model(img_width, img_height, vocab_size):
    input_img    = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    labels       = layers.Input(name="label",        shape=(None,), dtype="int64")
    input_length = layers.Input(name="input_length", shape=[1],     dtype="int64")
    label_length = layers.Input(name="label_length", shape=[1],     dtype="int64")

    # ── CNN Block ──────────────────────────────
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same", name="Conv1")(input_img)
    x = layers.MaxPooling2D((2,2), name="pool1")(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2,2), name="pool2")(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same", name="Conv3")(x)
    x = layers.MaxPooling2D((2,2), name="pool3")(x)

    # ── Reshape → Dense ────────────────────────
    # After 3× pool2d: width → W//8, height → H//8, channels → 128
    new_shape = ((img_width // 8), (img_height // 8) * 128)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # ── BiLSTM Block ───────────────────────────
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25), name="BiLSTM_1")(x)
    x = layers.Bidirectional(layers.LSTM(64,  return_sequences=True, dropout=0.25), name="BiLSTM_2")(x)

    # ── Output ─────────────────────────────────
    y_pred = layers.Dense(vocab_size + 1, activation="softmax", name="dense_output")(x)

    # ── CTC Loss ───────────────────────────────
    def ctc_loss_func(args):
        y_pred, _labels, _input_length, _label_length = args
        return tf.keras.backend.ctc_batch_cost(_labels, y_pred, _input_length, _label_length)

    loss_out = layers.Lambda(ctc_loss_func, output_shape=(1,), name="ctc_loss")(
        [y_pred, labels, input_length, label_length]
    )

    training_model   = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)
    prediction_model = Model(inputs=input_img, outputs=y_pred)

    return training_model, prediction_model


# ─────────────────────────────────────────────
# 6. CUSTOM METRICS (Character Error Rate)
# ─────────────────────────────────────────────

def character_error_rate(y_true_strs: list[str], y_pred_strs: list[str]) -> float:
    """
    CER = edit_distance(pred, true) / len(true)
    Averaged over the batch.
    """
    import editdistance  # pip install editdistance
    total_dist, total_len = 0, 0
    for true, pred in zip(y_true_strs, y_pred_strs):
        total_dist += editdistance.eval(pred, true)
        total_len  += max(len(true), 1)
    return total_dist / total_len


# ─────────────────────────────────────────────
# 7. CALLBACKS
# ─────────────────────────────────────────────

def get_callbacks(checkpoint_dir: str = "checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "ocr_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        TensorBoard(log_dir="logs", histogram_freq=0)
    ]


# ─────────────────────────────────────────────
# 8. TRAINING ENTRY POINT
# ─────────────────────────────────────────────

def train(dataset_dir: str,
          checkpoint_dir: str = "checkpoints",
          use_mjsynth: bool = False,
          use_char_folders: bool = False):
    """
    Main training function.

    Args:
        dataset_dir:    Path to your dataset folder.
        checkpoint_dir: Where to save model weights.
        use_mjsynth:    Set True if using MJSynth dataset layout.
        use_char_folders: Set True if using folder-per-character layout.
    """

    # ── Load data ─────────────────────────────
    if use_mjsynth:
        image_paths, labels = load_mjsynth(dataset_dir)
    elif use_char_folders:
        image_paths, labels = load_dataset_from_char_folders(dataset_dir)
    else:
        image_paths, labels = load_dataset_from_folder(dataset_dir)

    # Filter labels that contain out-of-vocabulary characters
    filtered = [
        (p, l) for p, l in zip(image_paths, labels)
        if all(ch in char_to_int for ch in l) and 0 < len(l) <= (IMG_WIDTH // 8)
    ]
    image_paths, labels = zip(*filtered)
    print(f"Samples after OOV filter: {len(image_paths)}")

    # ── Train / val split ─────────────────────
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.1, random_state=42
    )
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    # ── Build tf.data datasets ─────────────────
    train_ds = build_tf_dataset(train_paths, train_labels, augment=True,  shuffle=True)
    val_ds   = build_tf_dataset(val_paths,   val_labels,   augment=False, shuffle=False)

    # ── Build model ───────────────────────────
    training_model, prediction_model = build_ocr_model(IMG_WIDTH, IMG_HEIGHT, VOCAB_SIZE)

    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=lambda y_true, y_pred: y_pred   # CTC loss is already inside the graph
    )

    training_model.summary()

    # ── Train ─────────────────────────────────
    history = training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(checkpoint_dir)
    )

    # ── Save prediction model ──────────────────
    pred_model_path = os.path.join(checkpoint_dir, "ocr_predict.keras")
    prediction_model.save(pred_model_path)
    print(f"\nPrediction model saved to: {pred_model_path}")

    return history, prediction_model


# ─────────────────────────────────────────────
# 9. QUICK INFERENCE HELPER
# ─────────────────────────────────────────────

def predict_image(prediction_model, img_path: str) -> str:
    """Run inference on a single image and return decoded text."""
    img   = preprocess_image(img_path)
    img   = np.expand_dims(img, axis=0)        # add batch dim
    pred  = prediction_model.predict(img)       # (1, time_steps, vocab+1)
    return decode_prediction(pred[0])


# ─────────────────────────────────────────────
# 10. RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train OCR model")
    parser.add_argument("--dataset",    type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--checkpoints", type=str, default="checkpoints",
                        help="Directory to save model weights")
    parser.add_argument("--mjsynth",    action="store_true",
                        help="Use MJSynth dataset format")
    parser.add_argument("--char_folders", action="store_true",
                        help="Use folder-per-character dataset format")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        checkpoint_dir=args.checkpoints,
        use_mjsynth=args.mjsynth,
        use_char_folders=args.char_folders
    )
