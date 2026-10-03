"""
Training module for wafer defect classification.
Includes data loader, model factory, and two-phase training logic.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try importing EfficientNetV2
try:
    from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("EfficientNet not available. Please install tensorflow >= 2.6.0")

# Constants
DISPLAY_SIZE = (640, 640)
EFFICIENTNET_V2_SIZES = {'S': 384, 'M': 480}

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
class InteractiveWaferDefectLoader:
    """
    Loads wafer defect images on the fly using tf.data.Dataset.
    """
    def __init__(self, data_root, display_size=DISPLAY_SIZE):
        self.data_root = data_root
        self.display_size = display_size
        self.model_size = None
        self.label_encoder = LabelEncoder()
        self.data_root_path = self._find_data_path()
        self.all_images = self._scan_all_images()
        self.split_paths = {}
        self.split_labels = {}
        self._build_split_lists()
        self.label_encoder_fitted = False
        self.coord_tensor = None

    def _find_data_path(self):
        # Add custom path logic if needed
        if (os.path.exists(os.path.join(self.data_root, 'train')) and
            os.path.exists(os.path.join(self.data_root, 'valid')) and
            os.path.exists(os.path.join(self.data_root, 'test'))):
            return self.data_root
        # Fallback to common Kaggle structure
        candidate = os.path.join(self.data_root, 'Semiconductor_wafer')
        if os.path.exists(candidate):
            return candidate
        return self.data_root

    def _scan_all_images(self):
        splits = ['train', 'valid', 'test']
        catalog = {}
        for split in splits:
            split_path = os.path.join(self.data_root_path, split)
            if not os.path.exists(split_path):
                if split == 'valid':
                    split_path = os.path.join(self.data_root_path, 'val')
                if not os.path.exists(split_path):
                    continue
            catalog[split] = {}
            categories = [d for d in os.listdir(split_path)
                         if os.path.isdir(os.path.join(split_path, d))]
            for cat in categories:
                cat_path = os.path.join(split_path, cat)
                image_files = []
                for root, _, files in os.walk(cat_path):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                            image_files.append({
                                'path': os.path.join(root, f),
                                'filename': f,
                                'category': cat,
                                'split': split
                            })
                if image_files:
                    catalog[split][cat] = image_files
        return catalog

    def _build_split_lists(self):
        for split in ['train', 'valid', 'test']:
            if split not in self.all_images:
                self.split_paths[split] = []
                self.split_labels[split] = []
                continue
            paths = []
            labels = []
            for cat, file_infos in self.all_images[split].items():
                for info in file_infos:
                    paths.append(info['path'])
                    labels.append(cat)
            self.split_paths[split] = paths
            self.split_labels[split] = labels

    def fit_label_encoder(self):
        if 'train' not in self.split_labels:
            raise ValueError("No training data available to fit encoder.")
        self.label_encoder.fit(self.split_labels['train'])
        self.label_encoder_fitted = True
        print(f"Label encoder fitted. Classes: {self.label_encoder.classes_}")

    def set_model_size(self, size):
        self.model_size = (size, size)
        self.coord_tensor = self._generate_coord_tensor(self.model_size)

    def _generate_coord_tensor(self, target_size):
        h, w = target_size
        x = np.linspace(-1.0, 1.0, w)
        y = np.linspace(-1.0, 1.0, h)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        r = r / r.max()
        coord = np.stack([xx, yy, r], axis=-1).astype(np.float32)
        return tf.constant(coord)

    def _preprocess_image(self, path, label, augment=False):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, self.model_size, method='lanczos3')
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.concat([img, self.coord_tensor], axis=-1)
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    def get_dataset(self, split, batch_size, augment=False, shuffle=True, drop_remainder=True):
        if not self.label_encoder_fitted:
            raise RuntimeError("Must call fit_label_encoder() first.")
        if self.model_size is None:
            raise RuntimeError("model_size not set. Call set_model_size() first.")

        paths = self.split_paths[split]
        str_labels = self.split_labels[split]
        if len(paths) == 0:
            return None

        int_labels = self.label_encoder.transform(str_labels)
        num_classes = len(self.label_encoder.classes_)
        one_hot_labels = keras.utils.to_categorical(int_labels, num_classes).astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((paths, one_hot_labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(paths))

        dataset = dataset.map(
            lambda path, label: self._preprocess_image(path, label, augment),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

# -----------------------------------------------------------------------------
# Model Factory
# -----------------------------------------------------------------------------
class WaferDefectModelFactory:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _add_projection_if_needed(self, inputs):
        if self.input_shape[-1] == 3:
            return inputs
        x = layers.Conv2D(3, kernel_size=3, padding='same', use_bias=False,
                          name='coord_projection_conv')(inputs)
        x = layers.BatchNormalization(name='coord_projection_bn')(x)
        x = layers.Activation('relu', name='coord_projection_relu')(x)
        return x

    def build_efficientnet_v2(self, variant='S'):
        if not EFFICIENTNET_AVAILABLE:
            raise ImportError("EfficientNet not available")
        models_map = {'S': EfficientNetV2S, 'M': EfficientNetV2M}
        if variant not in models_map:
            variant = 'S'
        print(f"Building EfficientNetV2-{variant} with input {self.input_shape}")

        inputs = keras.Input(shape=self.input_shape)
        x = self._add_projection_if_needed(inputs)
        base = models_map[variant](
            include_top=False,
            weights='imagenet',
            input_shape=(self.input_shape[0], self.input_shape[1], 3),
            pooling='avg'
        )
        base.trainable = False

        x = base(x, training=False)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = layers.BatchNormalization(epsilon=1e-3)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = layers.BatchNormalization(epsilon=1e-3)(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        model = keras.Model(inputs, outputs)
        return model, base

# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def get_batch_size(num_gpus=1):
    return 4 * num_gpus

class TestSetCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, verbose=0):
        super().__init__()
        self.test_dataset = test_dataset
        self.verbose = verbose
        self.test_history = {
            'test_loss': [], 'test_accuracy': [],
            'test_precision': [], 'test_recall': [], 'test_auc': []
        }

    def on_epoch_end(self, epoch, logs=None):
        try:
            test_results = self.model.evaluate(self.test_dataset, verbose=self.verbose, return_dict=True)
            for key in self.test_history:
                if key[5:] in test_results:
                    self.test_history[key].append(test_results[key[5:]])
                    logs[key] = test_results[key[5:]]
        except Exception as e:
            print(f"Warning: Test evaluation failed at epoch {epoch}: {e}")

def train_wafer_model(data_loader, output_dir='./trained_models', variant='S'):
    """
    Train an EfficientNetV2 model (S or M) using tf.data.Dataset with coordinate channels.

    Args:
        data_loader: Instance of InteractiveWaferDefectLoader.
        output_dir: Directory to save models.
        variant: 'S' or 'M'.

    Returns:
        model, data_loader, variant, target_size
    """
    # Prepare data
    data_loader.fit_label_encoder()
    num_classes = len(data_loader.label_encoder.classes_)
    target_size = EFFICIENTNET_V2_SIZES[variant]
    print(f"Using EfficientNetV2-{variant} with input size {target_size}x{target_size}")

    variant_output_dir = os.path.join(output_dir, variant)
    os.makedirs(variant_output_dir, exist_ok=True)

    data_loader.set_model_size(target_size)
    input_shape = (target_size, target_size, 6)

    # Class weights
    train_labels = data_loader.split_labels['train']
    train_int_labels = data_loader.label_encoder.transform(train_labels)
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(train_int_labels),
                                         y=train_int_labels)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    # Batch size
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    batch_size = get_batch_size(num_gpus)

    # Datasets
    train_ds = data_loader.get_dataset('train', batch_size, augment=True, shuffle=True, drop_remainder=True)
    val_ds = data_loader.get_dataset('valid', batch_size, augment=False, shuffle=False, drop_remainder=True)
    test_ds = data_loader.get_dataset('test', batch_size, augment=False, shuffle=False, drop_remainder=True)

    if train_ds is None:
        raise ValueError("Training dataset is empty.")

    # Build model
    strategy = tf.distribute.MirroredStrategy() if num_gpus > 1 else tf.distribute.get_strategy()
    with strategy.scope():
        model_factory = WaferDefectModelFactory(input_shape, num_classes)
        model, base_model = model_factory.build_efficientnet_v2(variant)

        optimizer = keras.optimizers.Adam(learning_rate=5e-4, clipnorm=0.5)
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )

    # Callbacks
    best_model_path = os.path.join(variant_output_dir, 'best_model.keras')
    base_callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=80, restore_best_weights=True, mode='max', verbose=1),
        ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-7, verbose=1),
        TerminateOnNaN(),
    ]

    # Phase 1: Head only
    print("\n=== PHASE 1: Training classification head (frozen backbone) ===")
    base_model.trainable = False
    with strategy.scope():
        optimizer_phase1 = keras.optimizers.Adam(learning_rate=5e-4, clipnorm=0.5)
        model.compile(optimizer=optimizer_phase1,
                      loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                      metrics=['accuracy', 'precision', 'recall', 'auc'])
    phase1_callbacks = base_callbacks[:]
    if test_ds is not None:
        phase1_callbacks.append(TestSetCallback(test_ds, verbose=0))

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=phase1_callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Phase 2: Fine-tune last 40 layers
    print("\n=== PHASE 2: Partial fine-tuning (last 40 layers) ===")
    base_model.trainable = True
    # Freeze BN layers
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    total_layers = len(base_model.layers)
    for i, layer in enumerate(base_model.layers):
        if i < total_layers - 40:
            layer.trainable = False
        else:
            layer.trainable = True
    print(f"✓ Frozen first {total_layers - 40} layers, last 40 layers trainable.")

    with strategy.scope():
        optimizer_phase2 = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5)
        model.compile(optimizer=optimizer_phase2,
                      loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                      metrics=['accuracy', 'precision', 'recall', 'auc'])

    phase2_callbacks = base_callbacks[:]
    if test_ds is not None:
        test_callback_phase2 = TestSetCallback(test_ds, verbose=0)
        phase2_callbacks.insert(0, test_callback_phase2)

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=phase2_callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(variant_output_dir, 'final_model.keras')
    model.save(final_model_path)
    joblib.dump(data_loader.label_encoder, os.path.join(variant_output_dir, 'label_encoder.pkl'))
    with open(os.path.join(variant_output_dir, 'variant.txt'), 'w') as f:
        f.write(f"{variant}\n{target_size}")
    print(f"Model saved to {final_model_path}")

    return model, data_loader, variant, target_size

# -----------------------------------------------------------------------------
# Main (for direct execution)
# -----------------------------------------------------------------------------
def main():
    # Example usage
    data_root = input("Enter data root path: ").strip()
    loader = InteractiveWaferDefectLoader(data_root)
    loader.show_dataset_statistics()
    variant = input("Select variant (S or M): ").strip().upper()
    if variant not in ['S', 'M']:
        variant = 'S'
    model, loader, variant, target_size = train_wafer_model(loader, variant=variant)

if __name__ == "__main__":
    main()