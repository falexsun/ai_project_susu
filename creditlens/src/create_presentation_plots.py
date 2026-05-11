"""Генерация графиков для презентации / защиты проекта."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import CreditPreprocessor, DatasetConfig
from explainer import CreditExplainer
from model import CreditTrainer, get_processed_dir

# monkey-patch for joblib compat
import __main__
setattr(__main__, "CreditPreprocessor", CreditPreprocessor)
setattr(__main__, "DatasetConfig", DatasetConfig)

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

COLORS = {
    "primary": "#2563eb",
    "secondary": "#16a34a",
    "danger": "#dc2626",
    "warning": "#f59e0b",
    "purple": "#9333ea",
    "teal": "#14b8a6",
    "gray": "#94a3b8",
}

MODEL_REGISTRY = {
    "CreditNet (MLP)": "mlp_original",
    "CreditNet + Focal Loss": "mlp_focal",
    "Logistic Regression (C=0.1)": "logreg_c01",
    "Random Forest (depth=4)": "rf_depth4",
    "Gradient Boosting (depth=2)": "gb_depth2",
    "Stacking (MLP+RF+LogReg)": "stacking",
}


def _predict_with_model(model_artifact: dict, X: np.ndarray, all_models: dict | None = None) -> np.ndarray:
    model_type = model_artifact.get("type", "mlp")
    model = model_artifact["model"]
    if model_type == "mlp":
        return model.predict_proba(X)
    if model_type == "sklearn":
        base_labels = model_artifact.get("base_model_labels")
        if base_labels and all_models is not None:
            base_probs = []
            for label in base_labels:
                if label in all_models:
                    bp = all_models[label]["model"].predict_proba(X)
                    if bp.ndim == 2 and bp.shape[1] == 2:
                        bp = bp[:, 1]
                    base_probs.append(bp)
            if len(base_probs) >= 2:
                stacked = np.column_stack(base_probs)
                return model.predict_proba(stacked)[:, 1]
        return model.predict_proba(X)[:, 1]
    raise ValueError(f"Unknown model type: {model_type}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Дисбаланс классов
# ---------------------------------------------------------------------------
def plot_class_distribution(y_train: np.ndarray, y_test: np.ndarray, save_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Good (0)", "Bad (1)"]
    colors = [COLORS["secondary"], COLORS["danger"]]

    for ax, y, title in zip(axes, [y_train, y_test], ["Train", "Test"]):
        counts = [np.sum(y == 0), np.sum(y == 1)]
        bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=1.2)
        ax.set_title(f"{title}: {len(y)} samples")
        ax.set_ylabel("Количество")
        for bar, count in zip(bars, counts):
            pct = count / len(y) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(counts) * 1.2)

    plt.suptitle("Распределение классов в German Credit Dataset", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "01_class_distribution.png", bbox_inches="tight")
    plt.close()
    print("[plots] 01_class_distribution.png")


# ---------------------------------------------------------------------------
# 2. Корреляция признаков
# ---------------------------------------------------------------------------
def plot_correlation_matrix(df: pd.DataFrame, save_dir: Path) -> None:
    # Выбираем числовые признаки + target
    numeric_cols = ["Duration", "Amount", "InstallmentRate", "Residence", "Age",
                    "ExistingCredits", "Dependents", "Target"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Корреляционная матрица числовых признаков", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "02_correlation_matrix.png", bbox_inches="tight")
    plt.close()
    print("[plots] 02_correlation_matrix.png")


# ---------------------------------------------------------------------------
# 3. Duration vs Default Rate
# ---------------------------------------------------------------------------
def plot_duration_vs_default(df: pd.DataFrame, save_dir: Path) -> None:
    df["Default"] = (df["Target"] == 2).astype(int)
    bins = [0, 12, 18, 24, 36, 48, 60, 72]
    df["DurationBin"] = pd.cut(df["Duration"], bins=bins)
    grouped = df.groupby("DurationBin")["Default"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= 10]  # убираем шумные бины
    grouped["DurationBin"] = grouped["DurationBin"].astype(str)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    x_pos = np.arange(len(grouped))
    bars = ax1.bar(x_pos, grouped["mean"] * 100, color=COLORS["primary"],
                   edgecolor="black", linewidth=1.2, alpha=0.8)
    ax1.set_xlabel("Срок кредита, месяцы", fontsize=12)
    ax1.set_ylabel("Доля дефолтов, %", color=COLORS["primary"], fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(grouped["DurationBin"], rotation=30, ha="right")
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])
    ax1.set_ylim(0, 70)

    # Добавляем подписи на столбцы
    for bar, rate, count in zip(bars, grouped["mean"], grouped["count"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate*100:.1f}%\n(n={count})", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Линия тренда
    ax2 = ax1.twinx()
    ax2.plot(x_pos, grouped["mean"] * 100, color=COLORS["danger"], marker="o",
             linewidth=2.5, markersize=8, zorder=5)
    ax2.set_ylabel("Доля дефолтов, %", color=COLORS["danger"], fontsize=12)
    ax2.tick_params(axis="y", labelcolor=COLORS["danger"])
    ax2.set_ylim(0, 60)

    plt.title("Доля дефолтов по срокам кредита", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_dir / "03_duration_vs_default.png", bbox_inches="tight")
    plt.close()
    print("[plots] 03_duration_vs_default.png")


# ---------------------------------------------------------------------------
# 4. Кривые обучения CreditNet
# ---------------------------------------------------------------------------
def plot_training_curves(trainer: CreditTrainer, save_dir: Path) -> None:
    history = trainer.history
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(epochs, history["train_loss"], label="Train Loss", color=COLORS["primary"],
            linewidth=2.5, marker="o", markersize=5, markevery=3)
    ax.plot(epochs, history["val_loss"], label="Validation Loss", color=COLORS["danger"],
            linewidth=2.5, marker="s", markersize=5, markevery=3)

    # Отметка early stopping
    min_val_idx = np.argmin(history["val_loss"])
    ax.axvline(x=min_val_idx + 1, color=COLORS["warning"], linestyle="--",
               linewidth=2, label=f"Best val loss (epoch {min_val_idx + 1})")
    ax.scatter([min_val_idx + 1], [history["val_loss"][min_val_idx]],
               color=COLORS["warning"], s=150, zorder=5, edgecolors="black", linewidth=2)

    ax.set_xlabel("Эпоха", fontsize=12)
    ax.set_ylabel("Loss (BCEWithLogitsLoss)", fontsize=12)
    ax.set_title("Кривые обучения CreditNet", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0.5, len(epochs) + 0.5)

    # Аннотация — сдвинута левее, чтобы не обрезалась
    ax.annotate(f"Train: {history['train_loss'][-1]:.3f}\nVal: {history['val_loss'][-1]:.3f}",
                xy=(len(epochs), history["val_loss"][-1]),
                xytext=(len(epochs) - 7, history["val_loss"][-1] + 0.2),
                fontsize=10, color=COLORS["danger"],
                arrowprops=dict(arrowstyle="->", color=COLORS["danger"]))

    plt.tight_layout()
    plt.savefig(save_dir / "04_training_curves.png", bbox_inches="tight")
    plt.close()
    print("[plots] 04_training_curves.png")


# ---------------------------------------------------------------------------
# 5. Сравнение ROC-AUC (bar chart)
# ---------------------------------------------------------------------------
def plot_roc_auc_comparison(save_dir: Path) -> None:
    models = [
        "CreditNet\nOriginal",
        "CreditNet\nFocal Loss",
        "CreditNet\n5-Fold CV",
        "LogReg\n(C=0.1)",
        "Random Forest\n(depth=4)",
        "Gradient\nBoosting (d=2)",
        "Stacking\n(MLP+RF+LogReg)",
        "Decision\nTree (d=5)",
    ]
    roc_aucs = [0.8225, 0.8212, 0.8154, 0.8099, 0.7838, 0.7758, 0.8227, 0.6365]
    colors = [COLORS["primary"] if v > 0.80 else COLORS["warning"] if v > 0.75 else COLORS["danger"]
              for v in roc_aucs]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(models, roc_aucs, color=colors, edgecolor="black", linewidth=1.2, height=0.6)

    for bar, auc in zip(bars, roc_aucs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{auc:.4f}", va="center", fontsize=11, fontweight="bold")

    ax.axvline(x=0.8, color=COLORS["danger"], linestyle="--", linewidth=2,
               label="Целевой ROC-AUC = 0.8")
    ax.set_xlabel("ROC-AUC", fontsize=12)
    ax.set_title("Сравнение ROC-AUC всех моделей", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0.55, 0.85)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir / "05_roc_auc_comparison.png", bbox_inches="tight")
    plt.close()
    print("[plots] 05_roc_auc_comparison.png")


# ---------------------------------------------------------------------------
# 6. Сравнение переобучения (gap)
# ---------------------------------------------------------------------------
def plot_overfitting_gap(save_dir: Path) -> None:
    models = [
        "CreditNet\nOriginal",
        "CreditNet\nFocal Loss",
        "CreditNet\n5-Fold CV",
        "LogReg\n(C=0.1)",
        "Random Forest\n(depth=4)",
        "Gradient\nBoosting (d=2)",
        "Stacking\n(MLP+RF+LogReg)",
        "Decision\nTree (d=5)",
    ]
    gaps = [0.0485, 0.1038, 0.0874, 0.0171, 0.0680, 0.0954, 0.0504, 0.2023]
    colors = [COLORS["secondary"] if g < 0.03 else COLORS["warning"] if g < 0.07 else COLORS["danger"]
              for g in gaps]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(models, gaps, color=colors, edgecolor="black", linewidth=1.2, height=0.6)

    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{gap:.4f}", va="center", fontsize=11, fontweight="bold")

    ax.axvline(x=0.03, color=COLORS["secondary"], linestyle="--", linewidth=2,
               label="Низкое (< 0.03)")
    ax.axvline(x=0.07, color=COLORS["warning"], linestyle="--", linewidth=2,
               label="Умеренное (< 0.07)")
    ax.set_xlabel("Gap (Train ROC-AUC − Test ROC-AUC)", fontsize=12)
    ax.set_title("Переобучение моделей: gap train − test", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir / "06_overfitting_gap.png", bbox_inches="tight")
    plt.close()
    print("[plots] 06_overfitting_gap.png")


# ---------------------------------------------------------------------------
# 7. Сравнительные ROC-кривые (топ-5 моделей)
# ---------------------------------------------------------------------------
def plot_roc_curves_top5(X_test: np.ndarray, y_test: np.ndarray, all_models: dict, save_dir: Path) -> None:

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["purple"],
              COLORS["teal"], COLORS["warning"]]

    model_names = [
        "CreditNet (MLP)",
        "CreditNet + Focal Loss",
        "Logistic Regression (C=0.1)",
        "Stacking (MLP+RF+LogReg)",
        "Random Forest (depth=4)",
    ]

    for i, name in enumerate(model_names):
        if name not in all_models:
            continue
        artifact = all_models[name]
        probs = _predict_with_model(artifact, X_test, all_models)
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, lw=2.5, label=f"{name} (AUC={auc:.3f})", color=colors[i])

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Случайный (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Сравнительные ROC-кривые (топ-5 моделей)", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(save_dir / "07_roc_curves_top5.png", bbox_inches="tight")
    plt.close()
    print("[plots] 07_roc_curves_top5.png")


# ---------------------------------------------------------------------------
# 8. SHAP Waterfall для примера
# ---------------------------------------------------------------------------
def plot_shap_waterfall(trainer: CreditTrainer, X_train: np.ndarray, X_test: np.ndarray,
                        feature_names: list[str], preprocessor, save_dir: Path) -> None:
    explainer = CreditExplainer(trainer.model, X_train, feature_names)
    shap_info = explainer.explain(X_test[0:1])

    fig, ax = plt.subplots(figsize=(12, 8))

    shap_values = shap_info["shap_values"]
    prediction = shap_info["prediction"]

    # Топ-10 факторов
    top_idx = np.argsort(np.abs(shap_values))[-10:]
    vals = shap_values[top_idx]
    names = [feature_names[i] for i in top_idx]

    # Человекочитаемые имена
    from text_generator import _normalize_feature_name
    sample = shap_info.get("sample", np.zeros_like(shap_values))
    friendly_names = [_normalize_feature_name(n, sample[top_idx[j]] if len(sample) > top_idx[j] else None)
                      for j, n in enumerate(names)]

    colors_bar = [COLORS["danger"] if v > 0 else COLORS["secondary"] for v in vals]
    ax.barh(range(len(vals)), vals, color=colors_bar, edgecolor="black", linewidth=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(friendly_names, fontsize=10)
    ax.set_xlabel("Вклад SHAP (влияние на вероятность дефолта)", fontsize=12)
    ax.set_title(f"SHAP-объяснение решения (p={prediction:.3f})", fontsize=16, fontweight="bold")
    ax.axvline(0, color="black", linewidth=1.2)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Добавляем легенду
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["danger"], edgecolor="black", label="Увеличивает риск"),
        Patch(facecolor=COLORS["secondary"], edgecolor="black", label="Снижает риск"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir / "08_shap_waterfall.png", bbox_inches="tight")
    plt.close()
    print("[plots] 08_shap_waterfall.png")


# ---------------------------------------------------------------------------
# 9. Глобальная важность признаков (SHAP)
# ---------------------------------------------------------------------------
def plot_global_shap_importance(trainer: CreditTrainer, X_test: np.ndarray,
                                feature_names: list[str], save_dir: Path) -> None:
    explainer = CreditExplainer(trainer.model, X_test[:200], feature_names)
    shap_vals_raw = explainer.explainer.shap_values(
        __import__("torch", fromlist=["tensor"]).tensor(X_test[:200], dtype=__import__("torch").float32)
    )
    if isinstance(shap_vals_raw, list):
        shap_vals_raw = shap_vals_raw[0]
    shap_vals = np.array(shap_vals_raw)

    mean_abs = np.mean(np.abs(shap_vals), axis=0).flatten()
    order = np.argsort(mean_abs)[-15:].tolist()

    from text_generator import _normalize_feature_name
    friendly_names = [_normalize_feature_name(feature_names[int(i)]) for i in order]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_bar = [COLORS["primary"] if i % 2 == 0 else COLORS["teal"] for i in range(len(order))]
    ax.barh(friendly_names, mean_abs[order], color=colors_bar, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Среднее абсолютное значение SHAP", fontsize=12)
    ax.set_title("Глобальная важность признаков (топ-15)", fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_dir / "09_global_shap_importance.png", bbox_inches="tight")
    plt.close()
    print("[plots] 09_global_shap_importance.png")


# ---------------------------------------------------------------------------
# 10. What-if анализ (сумма и срок)
# ---------------------------------------------------------------------------
def plot_what_if(trainer: CreditTrainer, preprocessor, row_template: pd.DataFrame,
                 feature_names: list[str], save_dir: Path) -> None:
    # Варьируем сумму
    amounts_dm = range(2000, 12000, 500)
    probs_amount = []
    for amt in amounts_dm:
        r = row_template.copy()
        r["Amount"] = amt
        X = preprocessor.transform(r)
        probs_amount.append(trainer.predict_proba(X)[0])

    # Варьируем срок
    durations = range(6, 61, 6)
    probs_duration = []
    for dur in durations:
        r = row_template.copy()
        r["Duration"] = dur
        X = preprocessor.transform(r)
        probs_duration.append(trainer.predict_proba(X)[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Сумма
    axes[0].plot([a * 50 for a in amounts_dm], [p * 100 for p in probs_amount],
                 color=COLORS["primary"], linewidth=2.5, marker="o", markersize=6)
    axes[0].set_xlabel("Сумма кредита, ₽", fontsize=12)
    axes[0].set_ylabel("Вероятность дефолта, %", fontsize=12)
    axes[0].set_title("What-if: влияние суммы кредита", fontsize=14, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].axhline(y=50, color=COLORS["danger"], linestyle="--", linewidth=1.5, alpha=0.7)

    # Срок
    axes[1].plot(durations, [p * 100 for p in probs_duration],
                 color=COLORS["danger"], linewidth=2.5, marker="s", markersize=6)
    axes[1].set_xlabel("Срок кредита, месяцев", fontsize=12)
    axes[1].set_ylabel("Вероятность дефолта, %", fontsize=12)
    axes[1].set_title("What-if: влияние срока кредита", fontsize=14, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].axhline(y=50, color=COLORS["danger"], linestyle="--", linewidth=1.5, alpha=0.7)

    plt.suptitle("What-if анализ для типового клиента", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "10_what_if_analysis.png", bbox_inches="tight")
    plt.close()
    print("[plots] 10_what_if_analysis.png")


# ---------------------------------------------------------------------------
# 11. Precision-Recall trade-off
# ---------------------------------------------------------------------------
def plot_precision_recall_tradeoff(save_dir: Path) -> None:
    models = ["CreditNet\nOriginal", "CreditNet\nFocal Loss", "LogReg\n(C=0.1)"]
    precisions = [0.5385, 0.7083, 0.5281]
    recalls = [0.8167, 0.5667, 0.7833]
    colors = [COLORS["primary"], COLORS["purple"], COLORS["teal"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (model, p, r, c) in enumerate(zip(models, precisions, recalls, colors)):
        ax.scatter(r, p, s=400, color=c, edgecolors="black", linewidth=2, zorder=5)
        ax.annotate(model, (r, p), textcoords="offset points", xytext=(10, 10),
                    fontsize=11, fontweight="bold")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision vs Recall (три ключевые модели)", fontsize=16, fontweight="bold")
    ax.set_xlim(0.45, 0.88)
    ax.set_ylim(0.48, 0.78)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Легенда
    ax.text(0.46, 0.75, "→ Верхний правый угол = идеал", fontsize=10,
            color=COLORS["gray"], style="italic")

    plt.tight_layout()
    plt.savefig(save_dir / "11_precision_recall.png", bbox_inches="tight")
    plt.close()
    print("[plots] 11_precision_recall.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = get_processed_dir(project_root, "german")
    save_dir = ensure_dir(project_root / "data" / "plots" / "presentation")

    # Load data
    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    preprocessor = joblib.load(processed_dir / "preprocessor.pkl")
    feature_names = list(preprocessor.get_feature_names())
    trainer = CreditTrainer.load(processed_dir / "model.pt", device="cpu")

    df_raw = pd.read_csv(project_root / "data" / "raw" / "german_credit.csv")

    # Load all models for ROC comparison

    models_dir = processed_dir / "models"
    all_models = {}
    if models_dir.exists():
        for label, key in MODEL_REGISTRY.items():
            pkl_path = models_dir / f"{key}.pkl"
            if pkl_path.exists():
                all_models[label] = joblib.load(pkl_path)

    print("[plots] Генерация графиков для презентации...\n")

    plot_class_distribution(y_train, y_test, save_dir)
    plot_correlation_matrix(df_raw, save_dir)
    plot_duration_vs_default(df_raw, save_dir)
    plot_training_curves(trainer, save_dir)
    plot_roc_auc_comparison(save_dir)
    plot_overfitting_gap(save_dir)
    plot_roc_curves_top5(X_test, y_test, all_models, save_dir)
    plot_shap_waterfall(trainer, X_train, X_test, feature_names, preprocessor, save_dir)
    plot_global_shap_importance(trainer, X_test, feature_names, save_dir)

    # What-if row template
    row_template = pd.DataFrame([{
        "Status": "A11", "Duration": 36, "History": "A33", "Purpose": "A40",
        "Amount": 8000, "Savings": "A65", "Employment": "A72", "InstallmentRate": 4,
        "PersonalStatus": "A93", "Guarantors": "A101", "Residence": 2, "Property": "A124",
        "Age": 30, "OtherInstallments": "A143", "Housing": "A151", "ExistingCredits": 2,
        "Job": "A172", "Dependents": 1, "Phone": "A191", "Foreign": "A202"
    }])
    plot_what_if(trainer, preprocessor, row_template, feature_names, save_dir)
    plot_precision_recall_tradeoff(save_dir)

    print(f"\n[plots] Все графики сохранены в: {save_dir}")
    for f in sorted(save_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
