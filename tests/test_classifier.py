import os
import sys
import pathlib

# Ensure repository root is on sys.path when running tests directly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.classifier import IntentClassifier


def test_train_and_predict(tmp_path):
    dataset = os.path.join(os.path.dirname(__file__), "..", "agent", "intent_dataset.jsonl")
    model_path = tmp_path / "intent_test_model.joblib"

    clf = IntentClassifier(model_path=str(model_path))
    clf.train_from_jsonl(dataset)

    label, proba = clf.predict_proba("What is reinforcement learning?")
    assert label in {"explanatory", "discovery", "technical"}
    assert proba >= 0.4
