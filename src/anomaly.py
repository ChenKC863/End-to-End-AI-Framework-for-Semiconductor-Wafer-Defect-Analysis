"""
Anomaly detection module for wafer defect analysis.
Uses per-class PCA + Isolation Forest to detect outliers.
"""

import os
import sqlite3
import numpy as np
import tensorflow as tf
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

# Import preprocessing
from .preprocess import generate_coord_tensor, preprocess_image_tf

def safe_convert(value):
    """Convert numpy types to native Python types for SQLite."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value

def build_feature_extractor(model):
    """
    Build a feature extractor that outputs the backbone features (before classification head).

    Args:
        model: Trained Keras model.

    Returns:
        tf.keras.Model: Feature extractor.
    """
    # Assuming the backbone is the 5th layer (index 4) after projection layers
    # Adjust indices based on your model structure.
    x = model.layers[0].output
    for layer in model.layers[1:5]:  # projection layers + backbone
        x = layer(x)
    return tf.keras.Model(inputs=model.input, outputs=x)

class AnomalyDetector:
    """
    Per-class anomaly detector using PCA + Isolation Forest.
    """

    def __init__(self, model_dir: str, variant: str = 'S', n_components=32, contamination=0.01):
        """
        Initialize detector by loading model and preparing feature extractor.

        Args:
            model_dir (str): Path to the model directory (e.g., 'EfficientNetV2/model/S').
            variant (str): Model variant ('S' or 'M').
            n_components (int): Number of PCA components.
            contamination (float): Expected proportion of outliers.
        """
        self.variant = variant.upper()
        self.model_dir = model_dir
        self.n_components = n_components
        self.contamination = contamination

        # Load variant info
        variant_path = os.path.join(model_dir, 'variant.txt')
        with open(variant_path, 'r') as f:
            saved_variant = f.readline().strip()
            self.target_size = int(f.readline().strip())
        print(f"Loading {saved_variant} model (input size {self.target_size})")

        # Load label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        self.label_encoder = joblib.load(encoder_path)
        self.classes = self.label_encoder.classes_

        # Load Keras model and build feature extractor
        model_path = os.path.join(model_dir, 'best_model.keras')
        self.model = tf.keras.models.load_model(model_path)
        self.feature_extractor = build_feature_extractor(self.model)

        # Precompute coordinate tensor
        self.coord_tensor = generate_coord_tensor(self.target_size)

        # Placeholders for per-class models
        self.class_models = {}

    def extract_features(self, image_path: str):
        """
        Extract 1280‑D features from an image.

        Args:
            image_path (str): Path to the image.

        Returns:
            np.ndarray: Feature vector of shape (1280,).
        """
        img_tensor = preprocess_image_tf(image_path, self.target_size, self.coord_tensor)
        feats = self.feature_extractor.predict(img_tensor, verbose=0)
        return feats.squeeze()

    def predict_class(self, image_path: str):
        """
        Predict defect class and confidence.

        Args:
            image_path (str): Path to the image.

        Returns:
            tuple: (predicted_label, confidence)
        """
        img_tensor = preprocess_image_tf(image_path, self.target_size, self.coord_tensor)
        probs = self.model.predict(img_tensor, verbose=0)[0]
        idx = np.argmax(probs)
        label = self.label_encoder.inverse_transform([idx])[0]
        return label, probs[idx]

    def fit(self, data_root: str, save_models_path: str = None):
        """
        Train per-class PCA + Isolation Forest models using training data.

        Args:
            data_root (str): Root directory containing 'train' subfolder.
            save_models_path (str): Path to save trained models (optional).
        """
        train_dir = os.path.join(data_root, 'train')
        class_features = {cls: [] for cls in self.classes}

        print("Extracting training features...")
        for cls in tqdm(self.classes):
            cls_dir = os.path.join(train_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                path = os.path.join(cls_dir, fname)
                feats = self.extract_features(path)
                class_features[cls].append(feats)

        print("Training per-class models...")
        for cls, feats_list in class_features.items():
            if not feats_list:
                print(f"Warning: No features for class {cls}")
                continue
            X = np.array(feats_list)
            pca = PCA(n_components=self.n_components, random_state=42)
            X_pca = pca.fit_transform(X)
            iso = IsolationForest(
                n_estimators=300,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            iso.fit(X_pca)
            self.class_models[cls] = {'pca': pca, 'iso': iso}
            print(f"  {cls}: PCA variance = {pca.explained_variance_ratio_.sum():.4f}")

        if save_models_path:
            joblib.dump(self.class_models, save_models_path)
            print(f"Models saved to {save_models_path}")

    def detect(self, image_path: str, pred_label: str = None):
        """
        Detect if an image is an anomaly within its predicted class.

        Args:
            image_path (str): Path to the image.
            pred_label (str, optional): Predicted label. If None, will predict.

        Returns:
            tuple: (is_anomaly, anomaly_score)
        """
        if pred_label is None:
            pred_label, _ = self.predict_class(image_path)

        if pred_label not in self.class_models:
            raise ValueError(f"No model trained for class {pred_label}")

        feats = self.extract_features(image_path)
        model_pack = self.class_models[pred_label]
        feats_pca = model_pack['pca'].transform(feats.reshape(1, -1))
        score = model_pack['iso'].decision_function(feats_pca)[0]
        is_anomaly = int(model_pack['iso'].predict(feats_pca)[0] == -1)
        return is_anomaly, score

    def process_split(self, data_root: str, split: str, db_path: str):
        """
        Process all images in a split (train/valid/test) and store results in SQLite.

        Args:
            data_root (str): Root directory containing split folders.
            split (str): 'train', 'valid', or 'test'.
            db_path (str): Path to SQLite database.
        """
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"Split directory {split_dir} not found.")
            return

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS wafers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE,
                split TEXT,
                true_label TEXT,
                pred_label TEXT,
                pred_prob REAL,
                anomaly_score REAL,
                is_anomaly INTEGER
            )
        ''')
        conn.commit()

        for true_label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, true_label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(label_dir, fname)

                # Predict
                pred_label, pred_conf = self.predict_class(img_path)

                # Detect anomaly
                is_anomaly, score = self.detect(img_path, pred_label)

                # Insert into DB
                values = (img_path, split, true_label, pred_label, pred_conf, score, is_anomaly)
                safe_values = tuple(safe_convert(v) for v in values)
                try:
                    c.execute('''
                        INSERT INTO wafers
                        (image_path, split, true_label, pred_label, pred_prob, anomaly_score, is_anomaly)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', safe_values)
                    conn.commit()
                except sqlite3.IntegrityError:
                    pass  # skip duplicate

        conn.close()
        print(f"Processed split {split} into {db_path}")

    def load_models(self, models_path: str):
        """
        Load pre-trained per-class models.

        Args:
            models_path (str): Path to saved models (.pkl).
        """
        self.class_models = joblib.load(models_path)
        print(f"Loaded per-class models from {models_path}")

# -----------------------------------------------------------------------------
# Example usage (if run as a module from the project root)
# Run with: python -m src.anomaly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: train anomaly detector and process all splits
    # Assumes the script is run from the project root (where src/ is a subdirectory)
    model_dir = './model/S'          # or './EfficientNetV2/model/M'
    data_root = '.'                                 # contains train/, valid/, test/
    save_models_path = './model/S/classwise_models_S.pkl'   # output path for models
    db_path = './Database/wafer_features_classwise_S.db'     # output SQLite database

    detector = AnomalyDetector(model_dir=model_dir, variant='S')
    detector.fit(data_root=data_root, save_models_path=save_models_path)
    detector.process_split(data_root, 'train', db_path)
    detector.process_split(data_root, 'valid', db_path)
    detector.process_split(data_root, 'test', db_path)