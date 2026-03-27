from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from src.model import CreditTrainer, evaluate_binary_metrics, get_processed_dir
except ModuleNotFoundError:
    from model import CreditTrainer, evaluate_binary_metrics, get_processed_dir


def train_base_models(X_train: np.ndarray, y_train: np.ndarray) -> tuple[LogisticRegression, GradientBoostingClassifier]:
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(X_train, y_train)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = neg / max(pos, 1.0)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train, sample_weight=sample_weight)

    return logreg, gb


def predict_ensemble_proba(
    X: np.ndarray,
    logreg: LogisticRegression,
    gb: GradientBoostingClassifier,
    trainer: CreditTrainer,
    meta: LogisticRegression,
) -> np.ndarray:
    p_logreg = logreg.predict_proba(X)[:, 1]
    p_gb = gb.predict_proba(X)[:, 1]
    p_mlp = trainer.predict_proba(X)
    stacked = np.column_stack([p_logreg, p_gb, p_mlp])
    return meta.predict_proba(stacked)[:, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение ensemble-модели для кредитного скоринга")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "uci_credit_card", "give_me_some_credit", "home_credit"],
        help="Название датасета",
    )
    parser.add_argument("--mlp-epochs", type=int, default=40, help="Эпохи дообучения MLP при отсутствии model.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = get_processed_dir(project_root, args.dataset)

    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    logreg, gb = train_base_models(X_train, y_train)

    mlp_path = processed_dir / "model.pt"
    if mlp_path.exists():
        trainer = CreditTrainer.load(mlp_path, device="cpu")
    else:
        trainer = CreditTrainer(input_dim=X_train.shape[1], device="cpu")
        trainer.train(X_train, y_train, epochs=args.mlp_epochs, lr=0.001)
        trainer.save(mlp_path)

    p_logreg_train = logreg.predict_proba(X_train)[:, 1]
    p_gb_train = gb.predict_proba(X_train)[:, 1]
    p_mlp_train = trainer.predict_proba(X_train)
    X_meta_train = np.column_stack([p_logreg_train, p_gb_train, p_mlp_train])

    meta = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta.fit(X_meta_train, y_train)

    ensemble_prob = predict_ensemble_proba(X_test, logreg, gb, trainer, meta)
    metrics = evaluate_binary_metrics(y_test, ensemble_prob, threshold=trainer.threshold)

    model_artifact: dict[str, Any] = {
        "dataset": args.dataset,
        "model_type": "stacking-logreg-gb-mlp",
        "logreg": logreg,
        "gb": gb,
        "meta": meta,
        "threshold": float(getattr(trainer, "threshold", 0.5)),
        "recommended_threshold": float(getattr(trainer, "recommended_threshold", 0.5)),
        "base_model_order": ["logreg", "gb", "mlp"],
    }

    artifact_path = processed_dir / "ensemble.pkl"
    joblib.dump(model_artifact, artifact_path)

    metrics_path = processed_dir / "ensemble_metrics.json"
    metrics_payload = {
        "dataset": args.dataset,
        "metrics": metrics,
        "threshold": model_artifact["threshold"],
        "recommended_threshold": model_artifact["recommended_threshold"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Датасет: {args.dataset}")
    print(f"Ensemble сохранен: {artifact_path}")
    print(f"Метрики сохранены: {metrics_path}")
    print("Метрики ensemble на тесте:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"  threshold: {model_artifact['threshold']:.3f}")


if __name__ == "__main__":
    main()
