"""
Inference module for wafer defect classification.
Loads trained Keras model and label encoder, performs prediction.
"""

import os
import numpy as np
import tensorflow as tf
import joblib
from tensorflow import keras

# Import preprocessing functions
from .preprocess import preprocess_image_pil, generate_coord_tensor

class WaferDefectPredictor:
    """
    Predictor class for wafer defect classification.
    Supports both Keras and ONNX models.
    """

    def __init__(self, model_dir: str, variant: str = 'S'):
        """
        Initialize predictor with model and label encoder.

        Args:
            model_dir (str): Path to the model directory (e.g., 'model/S').
            variant (str): Model variant ('S' or 'M').
        """
        self.variant = variant.upper()
        self.model_dir = model_dir

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

        # Precompute coordinate tensor for TensorFlow preprocessing
        self.coord_tensor = generate_coord_tensor(self.target_size)

        # Load model
        model_path = os.path.join(model_dir, 'best_model.keras')
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully.")

    def predict(self, image_path: str):
        """
        Predict defect class for a single image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: (predicted_class, confidence)
        """
        # Preprocess
        input_tensor = preprocess_image_pil(image_path, self.target_size)

        # Predict
        predictions = self.model.predict(input_tensor, verbose=0)[0]
        idx = np.argmax(predictions)
        pred_class = self.classes[idx]
        confidence = predictions[idx]

        return pred_class, confidence

    def predict_batch(self, image_paths):
        """
        Predict for a batch of images.

        Args:
            image_paths (list): List of image paths.

        Returns:
            list: List of (predicted_class, confidence) tuples.
        """
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

# For backward compatibility: function-based prediction
def predict(image_path: str, model_dir: str = 'model/S'):
    """
    Quick prediction using default settings.

    Args:
        image_path (str): Path to the input image.
        model_dir (str): Path to the model directory.

    Returns:
        tuple: (predicted_class, confidence)
    """
    variant = 'S' if 'S' in model_dir else 'M'
    predictor = WaferDefectPredictor(model_dir, variant)
    return predictor.predict(image_path)