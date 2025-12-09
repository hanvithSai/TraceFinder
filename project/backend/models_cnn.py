class CNNScannerModel:
    """CNN model wrapper that lazy-loads TensorFlow/Keras and model files.

    This avoids import-time failures when TensorFlow or other compiled
    dependencies are not installed. Model and encoder are loaded on first use.
    """

    def __init__(self):
        self.model = None
        self.le = None

    def _load(self):
        if self.model is not None and self.le is not None:
            return
        try:
            import pickle
            from tensorflow import keras
            from config import CNN_MODEL_PATH, CNN_ENCODER_PATH
        except Exception as e:
            raise ImportError(f"Failed to import CNN dependencies: {e}") from e

        # load model and encoder
        self.model = keras.models.load_model(CNN_MODEL_PATH)
        with open(CNN_ENCODER_PATH, "rb") as f:
            self.le = pickle.load(f)

    def predict(self, img_path):
        try:
            # load heavy deps lazily
            self._load()
            from preprocessing_cnn import preprocess_image_cnn
            import numpy as np
        except Exception as e:
            raise RuntimeError(f"CNN model unavailable: {e}") from e

        x = preprocess_image_cnn(img_path)
        probs = self.model.predict(x)[0]
        idx = int(np.argmax(probs))
        label = self.le.inverse_transform([idx])[0]
        confidence = float(probs[idx])

        return {
            "model": "cnn",
            "label": label,
            "confidence": confidence,
            "probs": probs.tolist()
        }