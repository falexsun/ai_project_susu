"""
Скрипт сравнительного бенчмарка моделей кредитного скоринга.

Реализует единый пайплайн оценки:
- Logistic Regression (baseline)
- Gradient Boosting (tree-based)
- MLP CreditNet (нейронная сеть)
- Stacking Ensemble (meta-learning)

Результаты сохраняются в виде Markdown-таблицы и JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from model import CreditTrainer, evaluate_binary_metrics, get_processed_dir
from training_utils import compare_models_table, save_metrics_report

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Бенчмарк моделей кредитного скоринга")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "uci_credit_card", "give_me_some_credit", "home_credit"],
        help="Название датасета",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = get_processed_dir(project_root, args.dataset)
    plots_dir = project_root / "data" / "plots" / args.dataset
    plots_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    results: dict[str, dict[str, float]] = {}
    roc_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # 1. Logistic Regression
    print("[benchmark] Обучение LogisticRegression...")
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(X_train, y_train)
    prob_lr = logreg.predict_proba(X_test)[:, 1]
    metrics_lr = evaluate_binary_metrics(y_test, prob_lr)
    results["LogReg"] = metrics_lr
    roc_data["LogReg"] = (y_test, prob_lr)

    # 2. Gradient Boosting
    print("[benchmark] Обучение GradientBoosting...")
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    sample_weight = np.where(y_train == 1, neg / max(pos, 1.0), 1.0)
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train, sample_weight=sample_weight)
    prob_gb = gb.predict_proba(X_test)[:, 1]
    metrics_gb = evaluate_binary_metrics(y_test, prob_gb)
    results["GradientBoosting"] = metrics_gb
    roc_data["GradientBoosting"] = (y_test, prob_gb)

    # 3. MLP (CreditNet)
    print("[benchmark] Загрузка/обучение MLP (CreditNet)...")
    mlp_path = processed_dir / "model.pt"
    if mlp_path.exists():
        trainer = CreditTrainer.load(mlp_path, device="cpu")
    else:
        trainer = CreditTrainer(input_dim=X_train.shape[1], device="cpu")
        trainer.train(X_train, y_train, epochs=40, lr=0.001)
        trainer.save(mlp_path)
    prob_mlp = trainer.predict_proba(X_test)
    metrics_mlp = evaluate_binary_metrics(y_test, prob_mlp, threshold=trainer.threshold)
    results["MLP (CreditNet)"] = metrics_mlp
    roc_data["MLP (CreditNet)"] = (y_test, prob_mlp)

    # 4. Stacking Ensemble
    print("[benchmark] Обучение Stacking Ensemble...")
    p_lr_train = logreg.predict_proba(X_train)[:, 1]
    p_gb_train = gb.predict_proba(X_train)[:, 1]
    p_mlp_train = trainer.predict_proba(X_train)
    X_meta_train = np.column_stack([p_lr_train, p_gb_train, p_mlp_train])

    meta = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta.fit(X_meta_train, y_train)

    X_meta_test = np.column_stack([
        logreg.predict_proba(X_test)[:, 1],
        gb.predict_proba(X_test)[:, 1],
        trainer.predict_proba(X_test),
    ])
    prob_ens = meta.predict_proba(X_meta_test)[:, 1]
    metrics_ens = evaluate_binary_metrics(y_test, prob_ens, threshold=trainer.threshold)
    results["Stacking Ensemble"] = metrics_ens
    roc_data["Stacking Ensemble"] = (y_test, prob_ens)

    # Сравнительная ROC-кривая
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = {"LogReg": "#3498db", "GradientBoosting": "#2ecc71", "MLP (CreditNet)": "#e74c3c", "Stacking Ensemble": "#9b59b6"}
    for name, (yt, yp) in roc_data.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})", color=colors[name])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайный")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Сравнительная ROC-кривая")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[benchmark] Сравнительная ROC сохранена: {plots_dir / 'roc_comparison.png'}")

    # Сохранение результатов
    table_md = compare_models_table(
        results,
        save_path=plots_dir / "benchmark_table.md",
    )

    save_metrics_report(
        {},
        save_path=plots_dir / "benchmark.json",
        extra={"dataset": args.dataset, "results": results},
    )

    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print(f"{'='*60}")
    print(table_md)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
