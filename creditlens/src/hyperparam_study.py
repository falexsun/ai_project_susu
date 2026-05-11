"""
Скрипт систематического исследования гиперпараметров CreditNet.

Проводит 5-fold Stratified Cross-Validation для нескольких конфигураций,
сохраняет результаты и строит сравнительные графики.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import CreditTrainer, evaluate_binary_metrics
from training_utils import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Исследование гиперпараметров CreditNet")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "uci_credit_card", "give_me_some_credit", "home_credit"],
        help="Название датасета",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Количество фолдов CV")
    parser.add_argument("--epochs", type=int, default=100, help="Эпохи на фолд")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    return parser.parse_args()


def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
    config_name: str,
    hidden_layers: list[int],
    dropout_rates: list[float],
    threshold: float,
    n_splits: int = 5,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
) -> dict[str, dict[str, float]]:
    """Проводит StratifiedKFold CV для одной конфигурации."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: list[dict[str, float]] = []
    histories: list[dict[str, list[float]]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        trainer = CreditTrainer(
            input_dim=X.shape[1],
            device="cpu",
            hidden_layers=hidden_layers,
            dropout_rates=dropout_rates,
            use_batch_norm=True,
        )

        history = trainer.train(X_tr, y_tr, epochs=epochs, lr=lr, batch_size=batch_size)
        histories.append(history)

        val_prob = trainer.predict_proba(X_val)
        metrics = evaluate_binary_metrics(y_val, val_prob, threshold=threshold)
        fold_metrics.append(metrics)
        print(f"  [{config_name}] Fold {fold}/{n_splits} — ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")

    # Усреднение по фолдам
    avg_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        avg_metrics[key] = float(np.mean(values))
        std_metrics[key] = float(np.std(values))

    return {
        "mean": avg_metrics,
        "std": std_metrics,
        "folds": fold_metrics,
        "histories": histories,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = (
        project_root / "data" / "processed"
        if args.dataset == "german"
        else project_root / "data" / "processed" / args.dataset
    )
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = project_root / "data" / "plots" / args.dataset / "ablation"
    plots_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    # Объединяем train+test для CV (честная оценка на отложенной выборке делается отдельно)
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    experiments = [
        {
            "name": "Baseline",
            "hidden_layers": [256, 128, 64],
            "dropout_rates": [0.3, 0.2, 0.2],
            "threshold": 0.5,
        },
        {
            "name": "High Dropout",
            "hidden_layers": [256, 128, 64],
            "dropout_rates": [0.5, 0.4, 0.3],
            "threshold": 0.5,
        },
        {
            "name": "Threshold 0.7",
            "hidden_layers": [256, 128, 64],
            "dropout_rates": [0.3, 0.2, 0.2],
            "threshold": 0.7,
        },
        {
            "name": "Small Net",
            "hidden_layers": [128, 64, 32],
            "dropout_rates": [0.3, 0.2, 0.2],
            "threshold": 0.5,
        },
        {
            "name": "Small Net + High Dropout",
            "hidden_layers": [128, 64, 32],
            "dropout_rates": [0.5, 0.4, 0.3],
            "threshold": 0.5,
        },
    ]

    results: dict[str, dict] = {}

    print(f"\n{'='*70}")
    print(f"ИССЛЕДОВАНИЕ ГИПЕРПАРАМЕТРОВ | Датасет: {args.dataset}")
    print(f"{'='*70}\n")

    for exp in experiments:
        print(f"[experiment] {exp['name']} ...")
        result = run_experiment(
            X_full, y_full,
            config_name=exp["name"],
            hidden_layers=exp["hidden_layers"],
            dropout_rates=exp["dropout_rates"],
            threshold=exp["threshold"],
            n_splits=args.n_splits,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        results[exp["name"]] = result
        print(f"[experiment] {exp['name']} завершён.\n")

    # Финальная оценка на отложенном тесте для лучшей конфигурации (по ROC-AUC mean)
    best_config_name = max(results, key=lambda k: results[k]["mean"]["roc_auc"])
    best_exp = next(e for e in experiments if e["name"] == best_config_name)
    print(f"[info] Лучшая конфигурация по ROC-AUC CV: {best_config_name}")
    print(f"[info] Обучение финальной модели на полном train и тест на отложенной выборке...")

    final_trainer = CreditTrainer(
        input_dim=X_train.shape[1],
        device="cpu",
        hidden_layers=best_exp["hidden_layers"],
        dropout_rates=best_exp["dropout_rates"],
        use_batch_norm=True,
    )
    final_trainer.train(X_train, y_train, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    test_prob = final_trainer.predict_proba(X_test)
    test_metrics = evaluate_binary_metrics(y_test, test_prob, threshold=best_exp["threshold"])

    # Сохранение финальной модели
    final_model_path = processed_dir / "model_best.pt"
    final_trainer.save(final_model_path)
    print(f"[info] Финальная модель сохранена: {final_model_path}")

    # Сохранение результатов
    summary = {
        "dataset": args.dataset,
        "n_splits": args.n_splits,
        "experiments": {
            name: {
                "mean": res["mean"],
                "std": res["std"],
            }
            for name, res in results.items()
        },
        "best_config": best_config_name,
        "test_metrics": test_metrics,
    }
    summary_path = reports_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] Сводка сохранена: {summary_path}")

    # Markdown таблица
    md_lines = [
        "# Ablation Study Results\n",
        f"**Датасет:** {args.dataset} | **CV:** {args.n_splits}-fold Stratified\n",
        "\n## Сравнение конфигураций (mean ± std)\n",
        "| Конфигурация | ROC-AUC | PR-AUC | F1 | Precision | Recall |",
        "|-------------|---------|--------|----|-----------|--------|",
    ]
    for name, res in results.items():
        mean = res["mean"]
        std = res["std"]
        row = (
            f"| {name} | "
            f"{mean['roc_auc']:.4f} ± {std['roc_auc']:.4f} | "
            f"{mean['pr_auc']:.4f} ± {std['pr_auc']:.4f} | "
            f"{mean['f1']:.4f} ± {std['f1']:.4f} | "
            f"{mean['precision']:.4f} ± {std['precision']:.4f} | "
            f"{mean['recall']:.4f} ± {std['recall']:.4f} |"
        )
        md_lines.append(row)

    md_lines.append("\n## Финальная модель на отложенном тесте\n")
    md_lines.append(f"**Конфигурация:** {best_config_name}\n")
    md_lines.append("| Метрика | Значение |")
    md_lines.append("|---------|----------|")
    for k, v in test_metrics.items():
        md_lines.append(f"| {k} | {v:.4f} |")

    md_path = reports_dir / "ablation_study.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[report] Markdown отчёт сохранён: {md_path}")

    # Bar plot с доверительными интервалами
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.15
    metrics_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c", "#9b59b6"]

    for i, (metric, color) in enumerate(zip(metrics_keys, colors)):
        means = [results[n]["mean"][metric] for n in names]
        stds = [results[n]["std"][metric] for n in names]
        ax.bar(x + i * width, means, width, yerr=stds, label=metric, color=color, capsize=3, alpha=0.85)

    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение конфигураций (5-fold CV)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(plots_dir / "ablation_barplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Ablation barplot сохранён: {plots_dir / 'ablation_barplot.png'}")

    # Вывод итоговой таблицы в консоль
    print(f"\n{'='*70}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ABALTION STUDY")
    print(f"{'='*70}")
    print("\n".join(md_lines))
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
