"""
OCR Configuration - Lightweight version for inference only
No training dependencies (sklearn, etc.)
"""
import numpy as np

# Configuration
IMG_WIDTH = 128
IMG_HEIGHT = 32

# Characters the model recognizes
CHARACTERS = sorted(list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-()/"
))
VOCAB_SIZE = len(CHARACTERS)

# Mappings
char_to_int = {ch: i + 1 for i, ch in enumerate(CHARACTERS)}
int_to_char = {i + 1: ch for i, ch in enumerate(CHARACTERS)}


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


def build_ocr_model(img_width, img_height, vocab_size):
    """
    Build CRNN model architecture for inference.
    Returns: (training_model, prediction_model)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="int64")
    input_length = layers.Input(name="input_length", shape=[1], dtype="int64")
    label_length = layers.Input(name="label_length", shape=[1], dtype="int64")

    # CNN Block
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same", name="Conv1")(input_img)
    x = layers.MaxPooling2D((2,2), name="pool1")(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2,2), name="pool2")(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same", name="Conv3")(x)
    x = layers.MaxPooling2D((2,2), name="pool3")(x)

    # Reshape → Dense
    new_shape = ((img_width // 8), (img_height // 8) * 128)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # BiLSTM Block
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25), name="BiLSTM_1")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25), name="BiLSTM_2")(x)

    # Output
    y_pred = layers.Dense(vocab_size + 1, activation="softmax", name="dense_output")(x)

    # CTC Loss
    def ctc_loss_func(args):
        y_pred, _labels, _input_length, _label_length = args
        return tf.keras.backend.ctc_batch_cost(_labels, y_pred, _input_length, _label_length)

    loss_out = layers.Lambda(ctc_loss_func, output_shape=(1,), name="ctc_loss")(
        [y_pred, labels, input_length, label_length]
    )

    training_model = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)
    prediction_model = Model(inputs=input_img, outputs=y_pred)

    return training_model, prediction_model
