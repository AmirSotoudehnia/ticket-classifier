from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import joblib
import numpy as np

from src.utils.config import get_model_path, get_vectorizer_path, get_labelmap_path

@dataclass
class Prediction:
    label: str
    confidence: float

class TicketClassifier:
    def __init__(self):
        self.model = joblib.load(get_model_path())
        self.vectorizer = joblib.load(get_vectorizer_path())
        self.labels = joblib.load(get_labelmap_path())

    def predict_one(self, text: str) -> Prediction:
        X = self.vectorizer.transform([text])

        # اگر مدل proba داشته باشد از آن استفاده می‌کنیم
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            return Prediction(label=self.model.classes_[idx], confidence=float(proba[idx]))

        # fallback: decision_function → softmax-like
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            if scores.ndim == 1:
                scores = np.expand_dims(scores, axis=0)
            scores = scores[0]
            exps = np.exp(scores - np.max(scores))
            probs = exps / np.sum(exps)
            idx = int(np.argmax(probs))
            return Prediction(label=self.model.classes_[idx], confidence=float(probs[idx]))

        # آخرین fallback
        label = self.model.predict(X)[0]
        return Prediction(label=label, confidence=0.5)