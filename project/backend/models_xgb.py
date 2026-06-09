class XGBScannerModel:
    """XGBoost model wrapper that lazy-loads heavy dependencies and model files.

    This prevents import-time failures when compiled libraries (numpy/scikit-image)
    are not available. The actual model and encoder are loaded on first predict.
    """

    def __init__(self):
        self.model = None
        self.le = None

    def _load(self):
        if self.model is not None and self.le is not None:
            return
        try:
            import pickle
            # local imports that may require compiled packages
            from config import XGB_MODEL_PATH, XGB_ENCODER_PATH
        except Exception as e:
            raise ImportError(f"Failed to import XGBoost dependencies: {e}") from e

        with open(XGB_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(XGB_ENCODER_PATH, "rb") as f:
            self.le = pickle.load(f)

    def predict(self, img_path):
        try:
            # load model files and any missing deps lazily
            self._load()
            # import feature extractor and numpy only when needed
            from preprocessing_xgb import extract_features_xgb
            import numpy as np
        except Exception as e:
            # raise a runtime error so the API can convert it to an HTTP error
            raise RuntimeError(f"XGBoost model unavailable: {e}") from e

        # Get two feature versions: density=150 and density=300
        feat_150, feat_300 = extract_features_xgb(img_path)

        # Predict for 150 DPI
        probs_150 = self.model.predict_proba(feat_150.reshape(1, -1))[0]
        idx_150 = int(np.argmax(probs_150))
        label_150 = self.le.inverse_transform([idx_150])[0]

        # Predict for 300 DPI
        probs_300 = self.model.predict_proba(feat_300.reshape(1, -1))[0]
        idx_300 = int(np.argmax(probs_300))
        label_300 = self.le.inverse_transform([idx_300])[0]

        # Return both results
        return {
            "model": "xgboost",
            "150dpi": {
                "label": label_150,
                "confidence": float(probs_150[idx_150]),
                "probs": probs_150.tolist()
            },
            "300dpi": {
                "label": label_300,
                "confidence": float(probs_300[idx_300]),
                "probs": probs_300.tolist()
            }
        }