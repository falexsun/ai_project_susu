"""Обучает и сохраняет все модели для сравнения в UI."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments_v2 import FocalLoss, train_mlp_kfold, predict_kfold_proba
from model import CreditTrainer, get_processed_dir


def save_all_models(dataset: str = "german") -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = get_processed_dir(project_root, dataset)
    models_dir = processed_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    print(f"[save_models] Dataset: {dataset} | Train: {X_train.shape} | Test: {X_test.shape}")

    # 1. CreditNet Original (уже есть, но скопируем)
    print("[save_models] 1/6 CreditNet Original...")
    mlp_path = processed_dir / "model.pt"
    trainer_orig = CreditTrainer.load(mlp_path, device="cpu") if mlp_path.exists() else None
    if trainer_orig is None:
        trainer_orig = CreditTrainer(input_dim=X_train.shape[1], device="cpu")
        trainer_orig.train(X_train, y_train, epochs=100, lr=0.001)
        trainer_orig.save(mlp_path)
    joblib.dump({"model": trainer_orig, "type": "mlp"}, models_dir / "mlp_original.pkl")

    # 2. CreditNet + Focal Loss (5-fold, сохраняем первый трейнер как представителя)
    print("[save_models] 2/6 CreditNet + Focal Loss...")
    trainers_focal, _ = train_mlp_kfold(X_train, y_train, n_splits=5, epochs=100, lr=0.001, patience=15, use_focal=True)
    joblib.dump({"model": trainers_focal[0], "type": "mlp", "note": "focal_loss_fold0"}, models_dir / "mlp_focal.pkl")

    # 3. LogReg (C=0.1)
    print("[save_models] 3/6 LogReg (C=0.1)...")
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, penalty="l2", solver="lbfgs")
    logreg.fit(X_train, y_train)
    joblib.dump({"model": logreg, "type": "sklearn"}, models_dir / "logreg_c01.pkl")

    # 4. Random Forest (max_depth=4)
    print("[save_models] 4/6 Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_split=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    joblib.dump({"model": rf, "type": "sklearn"}, models_dir / "rf_depth4.pkl")

    # 5. Gradient Boosting (max_depth=2)
    print("[save_models] 5/6 Gradient Boosting...")
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    sample_weight = np.where(y_train == 1, neg / max(pos, 1.0), 1.0)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train, sample_weight=sample_weight)
    joblib.dump({"model": gb, "type": "sklearn"}, models_dir / "gb_depth2.pkl")

    # 6. Stacking (MLP + RF + LogReg)
    print("[save_models] 6/6 Stacking...")
    mlp_prob_train = trainer_orig.predict_proba(X_train)
    rf_prob_train = rf.predict_proba(X_train)[:, 1]
    lr_prob_train = logreg.predict_proba(X_train)[:, 1]
    X_meta_train = np.column_stack([mlp_prob_train, rf_prob_train, lr_prob_train])

    meta = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta.fit(X_meta_train, y_train)
    joblib.dump(
        {
            "model": meta,
            "type": "sklearn",
            "base_models": ["mlp", "rf", "logreg"],
            "base_model_labels": ["CreditNet (MLP)", "Random Forest (depth=4)", "Logistic Regression (C=0.1)"],
        },
        models_dir / "stacking.pkl",
    )

    print(f"\n[save_models] Все модели сохранены в: {models_dir}")
    for f in sorted(models_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    save_all_models()
