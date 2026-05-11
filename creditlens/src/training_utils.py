"""
Модуль утилит для обучения и визуализации нейронных сетей.

Содержит функции для:
- построения кривых обучения (loss curves)
- ROC-кривых и PR-кривых
- матриц ошибок (confusion matrix)
- сравнительных таблиц метрик

Используется в рамках курсового проекта по дисциплине
"Нейронные сети и искусственный интеллект".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: Path | None = None,
    title: str = "Кривые обучения",
) -> None:
    """Строит график train/val loss по эпохам.

    Args:
        history: Словарь с ключами 'train_loss' и 'val_loss'.
        save_path: Путь для сохранения графика.
        title: Заголовок графика.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, history["val_loss"], "r-s", label="Val Loss", markersize=4)

    ax.set_xlabel("Эпоха", fontsize=12)
    ax.set_ylabel("Loss (BCEWithLogits)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(1, len(epochs))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Кривые обучения сохранены: {save_path}")
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path | None = None,
    model_name: str = "MLP",
) -> None:
    """Строит ROC-кривую и вычисляет AUC.

    Args:
        y_true: Истинные метки.
        y_prob: Предсказанные вероятности.
        save_path: Путь для сохранения графика.
        model_name: Название модели для легенды.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайный классификатор")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC-кривая", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] ROC-кривая сохранена: {save_path}")
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path | None = None,
    model_name: str = "MLP",
) -> None:
    """Строит Precision-Recall кривую.

    Args:
        y_true: Истинные метки.
        y_prob: Предсказанные вероятности.
        save_path: Путь для сохранения графика.
        model_name: Название модели для легенды.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, "g-", lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")

    baseline = float(np.mean(y_true))
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Базовый уровень = {baseline:.3f}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall кривая", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] PR-кривая сохранена: {save_path}")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
    labels: list[str] | None = None,
) -> None:
    """Строит матрицу ошибок с процентами.

    Args:
        y_true: Истинные метки.
        y_pred: Предсказанные метки.
        save_path: Путь для сохранения графика.
        labels: Подписи классов.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if labels is None:
        labels = ["Одобрено (0)", "Дефолт (1)"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Количество"},
    )
    ax.set_xlabel("Предсказанный класс", fontsize=12)
    ax.set_ylabel("Истинный класс", fontsize=12)
    ax.set_title("Матрица ошибок", fontsize=14, fontweight="bold")

    # Добавляем проценты внутри ячеек
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5,
                i + 0.75,
                f"({cm_norm[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Матрица ошибок сохранена: {save_path}")
    plt.close(fig)


def save_metrics_report(
    metrics: dict[str, float],
    save_path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Сохраняет метрики в JSON для воспроизводимости.

    Args:
        metrics: Словарь метрик.
        save_path: Путь для сохранения.
        extra: Дополнительные поля (гиперпараметры и т.д.).
    """
    payload = {"metrics": metrics}
    if extra:
        payload["extra"] = extra
    save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] Метрики сохранены: {save_path}")


def compare_models_table(
    results: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> str:
    """Формирует Markdown-таблицу сравнения моделей.

    Args:
        results: {model_name: {metric: value}}.
        save_path: Если указан, сохраняет таблицу в MD.

    Returns:
        Строка Markdown-таблицы.
    """
    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    header = "| Модель | " + " | ".join(metrics) + " |"
    sep = "|" + "|".join(["-"] * (len(metrics) + 1)) + "|"
    rows = []
    for model in models:
        row = f"| {model} | " + " | ".join(f"{results[model][m]:.4f}" for m in metrics) + " |"
        rows.append(row)

    table = "\n".join([header, sep] + rows)
    if save_path:
        save_path.write_text(table, encoding="utf-8")
        print(f"[report] Таблица сравнения сохранена: {save_path}")
    return table


def plot_ablation_barplot(
    results: dict[str, dict[str, dict[str, float]]],
    save_path: Path | None = None,
    title: str = "Ablation Study (5-fold CV)",
) -> None:
    """Строит bar-plot сравнения конфигураций с error bars.

    Args:
        results: {config_name: {'mean': {metric: val}, 'std': {metric: val}}}.
        save_path: Путь для сохранения.
        title: Заголовок графика.
    """
    import matplotlib.pyplot as plt

    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.15
    metrics_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (metric, color) in enumerate(zip(metrics_keys, colors)):
        means = [results[n]["mean"][metric] for n in names]
        stds = [results[n]["std"][metric] for n in names]
        ax.bar(x + i * width, means, width, yerr=stds, label=metric, color=color, capsize=3, alpha=0.85)

    ax.set_ylabel("Значение метрики")
    ax.set_title(title)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Ablation barplot сохранён: {save_path}")
    plt.close(fig)
