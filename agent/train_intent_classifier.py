from __future__ import annotations

import argparse
import json
import sys
from agent.classifier import IntentClassifier


def main(dataset_path: str, model_path: str | None = None, eval_split: float = 0.2, report: bool = False, threshold: float = 0.75) -> None:
    clf = IntentClassifier(model_path=model_path)
    
    print(f"Training IntentClassifier on {dataset_path} with eval_split={eval_split}...")
    print(f"Using threshold={threshold} during inference")
    try:
        metrics = clf.train_from_jsonl(dataset_path, eval_split=eval_split)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    clf.metadata["threshold"] = threshold
    clf.save()

    print(f"Trained classifier saved to {clf.model_path}")
    
    if metrics:
        print(f"\nEvaluation Metrics (Accuracy: {metrics.get('accuracy', 0.0):.4f}):")
        if report and "report" in metrics:
            print(json.dumps(metrics["report"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train intent classifier from a JSONL dataset")
    parser.add_argument("dataset", help="Path to JSONL dataset with {\"text\", \"label\"}")
    parser.add_argument("--out", help="Optional output model path", default=None)
    parser.add_argument("--eval-split", type=float, default=0.2, help="Fraction of dataset to use for validation")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold for predictions")
    parser.add_argument("--report", action="store_true", help="Print detailed classification report")
    
    args = parser.parse_args()
    main(args.dataset, args.out, eval_split=args.eval_split, report=args.report, threshold=args.threshold)
