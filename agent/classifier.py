from __future__ import annotations

import json
import os
import time
from typing import Tuple, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


class IntentClassifier:
    """A TF-IDF + LogisticRegression intent router."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "intent_classifier.joblib")
        self.vectorizer: TfidfVectorizer | None = None
        self.model: LogisticRegression | None = None
        self.metadata: dict[str, Any] = {}
        if os.path.exists(self.model_path):
            self._load()

    def _load(self) -> None:
        data = joblib.load(self.model_path)
        self.vectorizer = data.get("vectorizer")
        self.model = data.get("model")
        self.metadata = data.get("metadata", {})

    def save(self) -> None:
        if not (self.vectorizer and self.model):
            raise RuntimeError("No trained model to save")
        joblib.dump({
            "vectorizer": self.vectorizer, 
            "model": self.model,
            "metadata": self.metadata
        }, self.model_path)

    def train_from_jsonl(self, path: str, eval_split: float = 0.2) -> dict[str, Any]:
        texts: list[str] = []
        labels: list[str] = []
        
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON at line {i}")
                    
                text = str(obj.get("text") or "").strip()
                label = str(obj.get("label") or "").strip().lower()
                
                if not text or not label:
                    raise ValueError(f"Missing text or label at line {i}")
                    
                texts.append(text)
                labels.append(label)

        if not texts:
            raise ValueError("No training examples found")

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        
        if eval_split > 0 and len(texts) > 5:
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=eval_split, random_state=42, stratify=labels)
        else:
            X_train, y_train = texts, labels
            X_test, y_test = [], []

        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        self.model.fit(X_train_vec, y_train)
        
        metrics = {}
        if X_test and y_test:
            metrics = self.evaluate(X_test, y_test)
            
        self.metadata = {
            "timestamp": time.time(),
            "dataset_path": path,
            "seed": 42,
            "labels": list(self.model.classes_),
            "vectorizer_params": self.vectorizer.get_params(),
            "metrics": metrics
        }
        self.save()
        return metrics

    def evaluate(self, texts: list[str], labels: list[str]) -> dict[str, Any]:
        if not (self.vectorizer and self.model):
            raise RuntimeError("Classifier not trained")
            
        X = self.vectorizer.transform(texts)
        y_pred = self.model.predict(X)
        
        return {
            "accuracy": float(accuracy_score(labels, y_pred)),
            "confusion_matrix": confusion_matrix(labels, y_pred).tolist(),
            "report": classification_report(labels, y_pred, output_dict=True, zero_division=0)
        }

    def predict_top_label(self, text: str) -> Tuple[str | None, float]:
        """Return the top label and its confidence probability."""
        if not text.strip():
            return None, 0.0
            
        if not (self.vectorizer and self.model):
            raise RuntimeError("Classifier not trained or model file missing")
            
        x = self.vectorizer.transform([text])
        probs = self.model.predict_proba(x)[0]
        idx = int(probs.argmax())
        return self.model.classes_[idx], float(probs[idx])

    def predict_proba(self, text: str) -> Tuple[str, float]:
        """Deprecated alias for predict_top_label."""
        label, prob = self.predict_top_label(text)
        return label or "discovery", prob

    def predict_with_threshold(self, text: str, threshold: float = 0.75) -> Tuple[str | None, float]:
        """Return (label, prob) if prob >= threshold, else (None, prob)."""
        if not text.strip():
            return None, 0.0
            
        label, prob = self.predict_top_label(text)
        if prob >= float(threshold):
            return label, prob
        return None, prob


__all__ = ["IntentClassifier"]
