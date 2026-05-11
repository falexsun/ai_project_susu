"""
Эксперименты v2: расширенный поиск лучшей модели.

Подходы:
1. CreditNet Original (baseline)
2. CreditNet + K-Fold CV ensemble (5 моделей, усреднение)
3. CreditNet + SMOTE (oversampling)
4. CreditNet + Focal Loss
5. CreditNet + patience=5 (ранняя остановка)
6. Logistic Regression (сильная L2 регуляризация)
7. Random Forest (max_depth=4)
8. Gradient Boosting (max_depth=2, lr=0.05)
9. Stacking ensemble (CV)
10. CreditNet + Polynomial Features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import CreditNet, CreditTrainer, evaluate_binary_metrics, get_processed_dir


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss для борьбы с дисбалансом и фокусировки на трудных примерах."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1.0 - p_t) ** self.gamma * bce_loss
        return loss.mean()


# ---------------------------------------------------------------------------
# K-Fold CV MLP
# ---------------------------------------------------------------------------
def train_mlp_kfold(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 15,
    use_focal: bool = False,
) -> tuple[list[CreditTrainer], list[float]]:
    """Обучает n_splits MLP моделей на кросс-валидации."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    trainers: list[CreditTrainer] = []
    val_aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  [CV fold {fold + 1}/{n_splits}]...")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        trainer = CreditTrainer(input_dim=X.shape[1], device="cpu")

        # Кастомное обучение с возможностью Focal Loss
        X_subtrain, X_subval, y_subtrain, y_subval = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
        )

        X_subtrain_t = torch.tensor(X_subtrain, dtype=torch.float32)
        y_subtrain_t = torch.tensor(y_subtrain.reshape(-1, 1), dtype=torch.float32)
        X_subval_t = torch.tensor(X_subval, dtype=torch.float32)
        y_subval_t = torch.tensor(y_subval.reshape(-1, 1), dtype=torch.float32)

        train_set = TensorDataset(X_subtrain_t, y_subtrain_t)
        val_set = TensorDataset(X_subval_t, y_subval_t)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        pos = float(y_tr.sum())
        neg = float(len(y_tr) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device="cpu")

        if use_focal:
            criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            trainer.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to("cpu"), yb.to("cpu")
                optimizer.zero_grad()
                logits = trainer.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_set)

            trainer.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to("cpu"), yb.to("cpu")
                    logits = trainer.model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_set)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        if best_state is not None:
            trainer.model.load_state_dict(best_state)

        val_prob = trainer.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, val_prob)
        val_aucs.append(val_auc)
        trainers.append(trainer)
        print(f"  [CV fold {fold + 1}] val_auc={val_auc:.4f}")

    return trainers, val_aucs


def predict_kfold_proba(trainers: list[CreditTrainer], X: np.ndarray) -> np.ndarray:
    probs = np.array([t.predict_proba(X) for t in trainers])
    return probs.mean(axis=0)


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------
def train_mlp_with_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 15,
) -> CreditTrainer:
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  [SMOTE] {X_train.shape} -> {X_res.shape}, balance={y_res.mean():.2%}")

    trainer = CreditTrainer(input_dim=X_res.shape[1], device="cpu")
    trainer.train(X_res, y_res, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience)
    return trainer


# ---------------------------------------------------------------------------
# Polynomial Features
# ---------------------------------------------------------------------------
def train_mlp_poly(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    degree: int = 2,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 15,
) -> tuple[CreditTrainer, Pipeline]:
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(poly.fit_transform(X_train))
    X_test_poly = scaler.transform(poly.transform(X_test))

    print(f"  [Poly] features: {X_train.shape[1]} -> {X_train_poly.shape[1]}")

    trainer = CreditTrainer(input_dim=X_train_poly.shape[1], device="cpu")
    trainer.train(X_train_poly, y_train, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience)
    return trainer, poly, scaler


# ---------------------------------------------------------------------------
# Sklearn модели
# ---------------------------------------------------------------------------
def train_logreg(X_train: np.ndarray, y_train: np.ndarray, C: float = 0.1) -> LogisticRegression:
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=C, penalty="l2", solver="lbfgs")
    lr.fit(X_train, y_train)
    return lr


def train_rf_limited(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_gb_limited(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    sample_weight = np.where(y_train == 1, neg / max(pos, 1.0), 1.0)
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        random_state=42,
    )
    gb.fit(X_train, y_train, sample_weight=sample_weight)
    return gb


# ---------------------------------------------------------------------------
# Оценка
# ---------------------------------------------------------------------------
def evaluate_sklearn(model, X_train, y_train, X_test, y_test, name: str) -> dict:
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    train_metrics = evaluate_binary_metrics(y_train, train_prob, threshold=0.5)
    test_metrics = evaluate_binary_metrics(y_test, test_prob, threshold=0.5)
    return {
        "name": name,
        "train": train_metrics,
        "test": test_metrics,
        "gap": {k: train_metrics[k] - test_metrics[k] for k in train_metrics},
        "test_prob": test_prob,
    }


def evaluate_mlp(trainer, X_train, y_train, X_test, y_test, name: str, poly_scaler=None) -> dict:
    if poly_scaler is not None:
        trainer, poly, scaler = poly_scaler
        X_train_eval = scaler.transform(poly.transform(X_train))
        X_test_eval = scaler.transform(poly.transform(X_test))
    else:
        X_train_eval = X_train
        X_test_eval = X_test

    train_prob = trainer.predict_proba(X_train_eval)
    test_prob = trainer.predict_proba(X_test_eval)
    train_metrics = evaluate_binary_metrics(y_train, train_prob, threshold=0.5)
    test_metrics = evaluate_binary_metrics(y_test, test_prob, threshold=0.5)
    return {
        "name": name,
        "train": train_metrics,
        "test": test_metrics,
        "gap": {k: train_metrics[k] - test_metrics[k] for k in train_metrics},
        "test_prob": test_prob,
    }


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def plot_roc_comparison(results: list[dict], y_test: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, res in enumerate(results):
        name = res["name"]
        fpr, tpr, _ = roc_curve(y_test, res["test_prob"])
        auc = roc_auc_score(y_test, res["test_prob"])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})", color=colors[i])

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайный")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Сравнительная ROC-кривая (эксперименты v2)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[experiments_v2] ROC сохранён: {save_path}")


def print_results_table(results: list[dict]) -> str:
    lines = []
    lines.append(f"{'Модель':<35} {'ROC-AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Gap':<10}")
    lines.append("-" * 90)
    for res in results:
        m = res["test"]
        g = res["gap"]["roc_auc"]
        lines.append(
            f"{res['name']:<35} {m['roc_auc']:<10.4f} {m['f1']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {g:+.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="german")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = get_processed_dir(project_root, args.dataset)
    plots_dir = project_root / "data" / "plots" / "experiments_v2"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    print(f"[experiments_v2] Dataset: {args.dataset} | Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[experiments_v2] Balance: {(y_train==1).mean():.1%} bad\n")

    results: list[dict] = []

    # 1. CreditNet Original (baseline)
    print("[1/10] CreditNet Original (baseline)...")
    mlp_path = processed_dir / "model.pt"
    if mlp_path.exists():
        trainer_orig = CreditTrainer.load(mlp_path, device="cpu")
    else:
        trainer_orig = CreditTrainer(input_dim=X_train.shape[1], device="cpu")
        trainer_orig.train(X_train, y_train, epochs=args.epochs, lr=args.lr)
    results.append(evaluate_mlp(trainer_orig, X_train, y_train, X_test, y_test, "1. CreditNet Original"))

    # 2. CreditNet + K-Fold CV (5 моделей)
    print("[2/10] CreditNet + 5-Fold CV Ensemble...")
    trainers_kfold, aucs_kfold = train_mlp_kfold(X_train, y_train, n_splits=5, epochs=args.epochs, lr=args.lr, patience=15)
    print(f"  [K-Fold] mean val auc={np.mean(aucs_kfold):.4f} (std={np.std(aucs_kfold):.4f})")
    test_prob_kfold = predict_kfold_proba(trainers_kfold, X_test)
    train_prob_kfold = predict_kfold_proba(trainers_kfold, X_train)
    train_metrics = evaluate_binary_metrics(y_train, train_prob_kfold, threshold=0.5)
    test_metrics = evaluate_binary_metrics(y_test, test_prob_kfold, threshold=0.5)
    results.append({
        "name": "2. CreditNet + 5-Fold CV",
        "train": train_metrics,
        "test": test_metrics,
        "gap": {k: train_metrics[k] - test_metrics[k] for k in train_metrics},
        "test_prob": test_prob_kfold,
    })

    # 3. CreditNet + SMOTE
    print("[3/10] CreditNet + SMOTE...")
    trainer_smote = train_mlp_with_smote(X_train, y_train, epochs=args.epochs, lr=args.lr)
    results.append(evaluate_mlp(trainer_smote, X_train, y_train, X_test, y_test, "3. CreditNet + SMOTE"))

    # 4. CreditNet + Focal Loss
    print("[4/10] CreditNet + Focal Loss...")
    trainers_focal, aucs_focal = train_mlp_kfold(X_train, y_train, n_splits=5, epochs=args.epochs, lr=args.lr, patience=15, use_focal=True)
    print(f"  [Focal] mean val auc={np.mean(aucs_focal):.4f} (std={np.std(aucs_focal):.4f})")
    test_prob_focal = predict_kfold_proba(trainers_focal, X_test)
    train_prob_focal = predict_kfold_proba(trainers_focal, X_train)
    train_metrics = evaluate_binary_metrics(y_train, train_prob_focal, threshold=0.5)
    test_metrics = evaluate_binary_metrics(y_test, test_prob_focal, threshold=0.5)
    results.append({
        "name": "4. CreditNet + Focal Loss",
        "train": train_metrics,
        "test": test_metrics,
        "gap": {k: train_metrics[k] - test_metrics[k] for k in train_metrics},
        "test_prob": test_prob_focal,
    })

    # 5. CreditNet + patience=5
    print("[5/10] CreditNet + patience=5...")
    trainer_p5 = CreditTrainer(input_dim=X_train.shape[1], device="cpu")
    trainer_p5.train(X_train, y_train, epochs=args.epochs, lr=args.lr, patience=5)
    results.append(evaluate_mlp(trainer_p5, X_train, y_train, X_test, y_test, "5. CreditNet + EarlyStop(p=5)"))

    # 6. Logistic Regression (сильная регуляризация)
    print("[6/10] Logistic Regression (C=0.1)...")
    lr_model = train_logreg(X_train, y_train, C=0.1)
    results.append(evaluate_sklearn(lr_model, X_train, y_train, X_test, y_test, "6. LogReg (C=0.1)"))

    # 7. Random Forest (max_depth=4)
    print("[7/10] Random Forest (max_depth=4)...")
    rf_model = train_rf_limited(X_train, y_train)
    results.append(evaluate_sklearn(rf_model, X_train, y_train, X_test, y_test, "7. RF (max_depth=4)"))

    # 8. Gradient Boosting (max_depth=2)
    print("[8/10] Gradient Boosting (max_depth=2)...")
    gb_model = train_gb_limited(X_train, y_train)
    results.append(evaluate_sklearn(gb_model, X_train, y_train, X_test, y_test, "8. GB (max_depth=2)"))

    # 9. Stacking: MLP + RF + LogReg
    print("[9/10] Stacking Ensemble...")
    mlp_prob_train = trainer_orig.predict_proba(X_train)
    rf_prob_train = rf_model.predict_proba(X_train)[:, 1]
    lr_prob_train = lr_model.predict_proba(X_train)[:, 1]
    X_meta_train = np.column_stack([mlp_prob_train, rf_prob_train, lr_prob_train])

    mlp_prob_test = trainer_orig.predict_proba(X_test)
    rf_prob_test = rf_model.predict_proba(X_test)[:, 1]
    lr_prob_test = lr_model.predict_proba(X_test)[:, 1]
    X_meta_test = np.column_stack([mlp_prob_test, rf_prob_test, lr_prob_test])

    meta = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta.fit(X_meta_train, y_train)
    meta_prob_train = meta.predict_proba(X_meta_train)[:, 1]
    meta_prob_test = meta.predict_proba(X_meta_test)[:, 1]
    train_metrics = evaluate_binary_metrics(y_train, meta_prob_train, threshold=0.5)
    test_metrics = evaluate_binary_metrics(y_test, meta_prob_test, threshold=0.5)
    results.append({
        "name": "9. Stacking (MLP+RF+LogReg)",
        "train": train_metrics,
        "test": test_metrics,
        "gap": {k: train_metrics[k] - test_metrics[k] for k in train_metrics},
        "test_prob": meta_prob_test,
    })

    # 10. CreditNet + Polynomial Features
    print("[10/10] CreditNet + Polynomial Features...")
    trainer_poly, poly, scaler = train_mlp_poly(X_train, y_train, X_test, degree=2, epochs=args.epochs, lr=args.lr)
    results.append(evaluate_mlp(trainer_poly, X_train, y_train, X_test, y_test, "10. CreditNet + PolyFeatures", poly_scaler=(trainer_poly, poly, scaler)))

    # -----------------------------------------------------------------------
    # Вывод
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ V2")
    print("=" * 90)
    table = print_results_table(results)
    print(table)

    plot_roc_comparison(results, y_test, plots_dir / "roc_experiments_v2.png")

    # Анализ переобучения
    print("\n" + "=" * 90)
    print("АНАЛИЗ ПЕРЕОБУЧЕНИЯ")
    print("=" * 90)
    for res in results:
        gap_auc = res["gap"]["roc_auc"]
        status = "🔴 Сильное" if gap_auc > 0.05 else "🟡 Умеренное" if gap_auc > 0.02 else "🟢 Низкое"
        print(f"{res['name']:<35} gap={gap_auc:+.4f}  {status}")

    # Лучшие модели
    best_auc = max(results, key=lambda r: r["test"]["roc_auc"])
    best_f1 = max(results, key=lambda r: r["test"]["f1"])
    print(f"\n🏆 Лучшая по ROC-AUC: {best_auc['name']} ({best_auc['test']['roc_auc']:.4f})")
    print(f"🏆 Лучшая по F1:      {best_f1['name']} ({best_f1['test']['f1']:.4f})")

    # Сохранение
    report = {
        "dataset": args.dataset,
        "models": [
            {"name": r["name"], "train_metrics": r["train"], "test_metrics": r["test"], "gap": {k: float(v) for k, v in r["gap"].items()}}
            for r in results
        ],
    }
    report_path = reports_dir / "experiments_v2_results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[experiments_v2] Отчёт: {report_path}")

    md_lines = ["# Результаты экспериментов v2\n", "| # | Модель | ROC-AUC | F1 | Precision | Recall | Gap |", "|---|--------|---------|-----|-----------|--------|-----|"]
    for i, r in enumerate(results, 1):
        m = r["test"]
        g = r["gap"]["roc_auc"]
        md_lines.append(f"| {i} | {r['name']} | {m['roc_auc']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {g:+.4f} |")
    md_path = reports_dir / "experiments_v2_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[experiments_v2] Таблица: {md_path}")


if __name__ == "__main__":
    main()
