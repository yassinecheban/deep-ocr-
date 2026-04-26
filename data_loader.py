import os
import random
import tensorflow as tf
from tensorflow.keras import layers

class OCRDataLoader:
    def __init__(self, dataset_path, img_width=128, img_height=32, batch_size=64):
        self.dataset_path = dataset_path
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        
        # Character set: 0-9, a-z, A-Z, and punctuation (66 characters total)
        self.characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-,./"
        
        # Folder to character mapping
        self.folder_map = {
            "comma": ",",
            "dash": "-",
            "dot": ".",
            "slash": "/"
        }
        # Add A_U to Z_U and a_L to z_L
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.folder_map[f"{c}_U"] = c
        for c in "abcdefghijklmnopqrstuvwxyz":
            self.folder_map[f"{c}_L"] = c
        # Numbers are handled directly
        for i in range(10):
            self.folder_map[str(i)] = str(i)

        # Character to Integer mapping
        self.char_to_num = layers.StringLookup(
            vocabulary=list(self.characters), mask_token=None
        )
        
        # Integer to Character mapping
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

    def _preprocess_image(self, img_path, label, bbox=None):
        try:
            # 1. Read and decode image
            img = tf.io.read_file(img_path)
            img = tf.io.decode_image(img, channels=1, expand_animations=False)
            
            # 2. Crop if bbox is provided [xtl, ytl, w, h]
            if bbox is not None:
                # Convert string representation of list to actual floats if needed
                if isinstance(bbox, str):
                    bbox = [float(x) for x in bbox.strip('[]').split(',')]
                
                xtl, ytl, w, h = tf.cast(bbox[0], tf.int32), tf.cast(bbox[1], tf.int32), \
                                tf.cast(bbox[2], tf.int32), tf.cast(bbox[3], tf.int32)
                
                # Ensure coordinates are within image bounds
                img_shape = tf.shape(img)
                ytl = tf.clip_by_value(ytl, 0, img_shape[0] - 1)
                xtl = tf.clip_by_value(xtl, 0, img_shape[1] - 1)
                h = tf.clip_by_value(h, 1, img_shape[0] - ytl)
                w = tf.clip_by_value(w, 1, img_shape[1] - xtl)
                
                img = tf.image.crop_to_bounding_box(img, ytl, xtl, h, w)

            # 3. Normalize and resize
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [self.img_height, self.img_width])
            
            # 4. Transpose for CTC
            img = tf.transpose(img, perm=[1, 0, 2])
            
            # 5. Map label characters to integers
            label_encoded = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
            
            # 6. Lengths for CTC Loss
            input_length = tf.cast(self.img_width // 8, tf.int64)
            label_length = tf.cast(tf.shape(label_encoded)[0], tf.int64)
            
            return {
                "image": img,
                "label": tf.cast(label_encoded, tf.int64),
                "input_length": [input_length],
                "label_length": [label_length]
            }, tf.cast(label_encoded, tf.int64)
        except Exception as e:
            dummy_img = tf.zeros([self.img_width, self.img_height, 1], dtype=tf.float32)
            dummy_label = self.char_to_num(tf.constant(["0"]))
            return {
                "image": dummy_img,
                "label": tf.cast(dummy_label, tf.int64),
                "input_length": [tf.cast(self.img_width // 8, tf.int64)],
                "label_length": [tf.cast(1, tf.int64)]
            }, tf.cast(dummy_label, tf.int64)

    def get_textocr_dataset(self, csv_path, img_dir, split_ratio=0.8, max_samples=50000):
        import pandas as pd
        print(f"Loading annotations from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Basic filtering: remove very short labels or nan
        df = df[df['utf8_string'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
        
        # Limit dataset size for performance if requested
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            
        image_paths = [os.path.join(img_dir, f"{img_id}.jpg") for img_id in df['image_id']]
        labels = df['utf8_string'].tolist()
        bboxes = df['bbox'].tolist()

        # Shuffle and Split
        combined = list(zip(image_paths, labels, bboxes))
        random.shuffle(combined)
        
        split_idx = int(len(combined) * split_ratio)
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]
        
        print(f"Prepared {len(combined)} samples. Train: {len(train_data)}, Val: {len(val_data)}")
        
        def create_tf_dataset(data):
            paths, lbls, bxs = zip(*data)
            ds = tf.data.Dataset.from_tensor_slices((list(paths), list(lbls), list(bxs)))
            ds = ds.shuffle(buffer_size=min(10000, len(data)))
            
            # Map with cropping support
            ds = ds.map(lambda p, l, b: self._preprocess_image(p, l, b), num_parallel_calls=tf.data.AUTOTUNE)
            
            ds = ds.padded_batch(
                self.batch_size,
                padded_shapes=(
                    {
                        "image": [self.img_width, self.img_height, 1],
                        "label": [None],
                        "input_length": [1],
                        "label_length": [1]
                    },
                    [None]
                ),
                padding_values=(
                    {
                        "image": 0.0,
                        "label": tf.cast(-1, tf.int64),
                        "input_length": tf.cast(0, tf.int64),
                        "label_length": tf.cast(0, tf.int64)
                    },
                    tf.cast(-1, tf.int64)
                ),
                drop_remainder=True
            )
            return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return create_tf_dataset(train_data), create_tf_dataset(val_data)

    def get_folder_dataset(self, split_ratio=0.8):
        image_paths = []
        labels = []
        
        print(f"Scanning directory: {self.dataset_path}")
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            char = self.folder_map.get(folder_name)
            if char is None:
                print(f"Warning: Unknown folder name '{folder_name}', skipping.")
                continue
                
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(folder_path, img_name))
                    labels.append(char)
        
        # Shuffle and Split
        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        
        split_idx = int(len(combined) * split_ratio)
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]
        
        print(f"Found {len(combined)} images. Train: {len(train_data)}, Val: {len(val_data)}")
        
        def create_tf_dataset(data):
            paths, lbls = zip(*data)
            ds = tf.data.Dataset.from_tensor_slices((list(paths), list(lbls)))
            
            # Shuffle at the beginning of each epoch
            ds = ds.shuffle(buffer_size=10000)
            
            ds = ds.map(self._preprocess_image, num_parallel_calls=4)
            
            ds = ds.padded_batch(
                self.batch_size,
                padded_shapes=(
                    {
                        "image": [self.img_width, self.img_height, 1],
                        "label": [None],
                        "input_length": [1],
                        "label_length": [1]
                    },
                    [None]
                ),
                padding_values=(
                    {
                        "image": 0.0,
                        "label": tf.cast(-1, tf.int64),
                        "input_length": tf.cast(0, tf.int64),
                        "label_length": tf.cast(0, tf.int64)
                    },
                    tf.cast(-1, tf.int64)
                ),
                drop_remainder=True
            )
            return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return create_tf_dataset(train_data), create_tf_dataset(val_data)
