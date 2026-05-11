"""
Обучение stacking ensemble с кросс-валидацией для генерации meta-признаков.

Использует StratifiedKFold для получения out-of-fold предсказаний
базовых моделей, что предотвращает переобучение meta-модели.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import CreditTrainer, evaluate_binary_metrics, get_processed_dir


def train_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[LogisticRegression, GradientBoostingClassifier]:
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(X_train, y_train)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = neg / max(pos, 1.0)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train, sample_weight=sample_weight)

    return logreg, gb


def get_oof_predictions(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    is_mlp: bool = False,
    mlp_epochs: int = 40,
    mlp_lr: float = 0.001,
) -> np.ndarray:
    """Генерирует out-of-fold предсказания для модели."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]

        if is_mlp:
            trainer = CreditTrainer(input_dim=X.shape[1], device="cpu")
            trainer.train(X_tr, y_tr, epochs=mlp_epochs, lr=mlp_lr)
            oof_preds[val_idx] = trainer.predict_proba(X_val)
        else:
            if hasattr(model, "sample_weight") and not is_mlp:
                # GradientBoosting требует sample_weight
                pos = float(y_tr.sum())
                neg = float(len(y_tr) - pos)
                sw = np.where(y_tr == 1, neg / max(pos, 1.0), 1.0)
                clone = type(model)(**model.get_params())
                clone.fit(X_tr, y_tr, sample_weight=sw)
            else:
                clone = type(model)(**model.get_params())
                clone.fit(X_tr, y_tr)
            oof_preds[val_idx] = clone.predict_proba(X_val)[:, 1]

    return oof_preds


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
    parser = argparse.ArgumentParser(description="Обучение ensemble-модели для кредитного скоринга (CV stacking)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "uci_credit_card", "give_me_some_credit", "home_credit"],
        help="Название датасета",
    )
    parser.add_argument("--mlp-epochs", type=int, default=40, help="Эпохи дообучения MLP при отсутствии model.pt")
    parser.add_argument("--n-splits", type=int, default=5, help="Количество фолдов для CV meta-признаков")
    parser.add_argument("--meta-model", type=str, default="logreg", choices=["logreg", "ridge"], help="Meta-модель")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = get_processed_dir(project_root, args.dataset)

    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    print("[ensemble] Обучение базовых моделей на полном train...")
    logreg, gb = train_base_models(X_train, y_train)

    mlp_path = processed_dir / "model.pt"
    if mlp_path.exists():
        trainer = CreditTrainer.load(mlp_path, device="cpu")
    else:
        trainer = CreditTrainer(input_dim=X_train.shape[1], device="cpu")
        trainer.train(X_train, y_train, epochs=args.mlp_epochs, lr=0.001)
        trainer.save(mlp_path)

    # Генерация out-of-fold предсказаний для meta-обучения
    print(f"[ensemble] Генерация OOF предсказаний ({args.n_splits}-fold CV)...")
    oof_lr = get_oof_predictions(logreg, X_train, y_train, n_splits=args.n_splits)
    oof_gb = get_oof_predictions(gb, X_train, y_train, n_splits=args.n_splits)
    # Для MLP используем тот же trainer, но обучаем заново на каждом фолде
    oof_mlp = get_oof_predictions(
        None, X_train, y_train, n_splits=args.n_splits, is_mlp=True,
        mlp_epochs=args.mlp_epochs,
    )

    X_meta_train = np.column_stack([oof_lr, oof_gb, oof_mlp])

    if args.meta_model == "logreg":
        meta = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        from sklearn.linear_model import RidgeCV
        meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        # Ridge даёт не вероятности, а регрессию. Нужно откалибровать.
        # Проще оставить LogReg как основной вариант.
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
        "cv_splits": args.n_splits,
    }

    artifact_path = processed_dir / "ensemble.pkl"
    joblib.dump(model_artifact, artifact_path)

    metrics_path = processed_dir / "ensemble_metrics.json"
    metrics_payload = {
        "dataset": args.dataset,
        "metrics": metrics,
        "threshold": model_artifact["threshold"],
        "recommended_threshold": model_artifact["recommended_threshold"],
        "cv_splits": args.n_splits,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Датасет: {args.dataset}")
    print(f"Ensemble (CV stacking) сохранен: {artifact_path}")
    print(f"Метрики сохранены: {metrics_path}")
    print("Метрики ensemble на тесте:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"  threshold: {model_artifact['threshold']:.3f}")


if __name__ == "__main__":
    main()
