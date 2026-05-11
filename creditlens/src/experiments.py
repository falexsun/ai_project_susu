"""
Эксперименты: борьба с переобучением и сравнение моделей.

Модели:
1. CreditNet Original  — baseline MLP (256→128→64, dropout 0.3/0.2/0.2)
2. CreditNet Regularized — уменьшенная MLP (128→64→32, dropout 0.5/0.3/0.3)
3. Random Forest — деревья с ограничением глубины
4. Gradient Boosting — уже есть в проекте
5. Decision Tree — одиночное дерево (ожидается переобучение)
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import CreditNet, CreditTrainer, evaluate_binary_metrics, get_processed_dir


# ---------------------------------------------------------------------------
# 1. CreditNet Regularized
# ---------------------------------------------------------------------------
class CreditNetReg(nn.Module):
    """Регуляризованная версия CreditNet с меньшей архитектурой."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | None = None,
        dropout_rates: list[float] | None = None,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [128, 64, 32]
        dropout_rates = dropout_rates or [0.5, 0.3, 0.3]

        if len(dropout_rates) != len(hidden_layers):
            raise ValueError("dropout_rates и hidden_layers должны иметь одинаковую длину")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim, dropout_p in zip(hidden_layers, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CreditTrainerReg:
    """Регуляризованный трейнер с усиленной регуляризацией."""

    def __init__(
        self,
        input_dim: int,
        device: str | None = None,
        hidden_layers: list[int] | None = None,
        dropout_rates: list[float] | None = None,
        use_batch_norm: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.arch_config = {
            "input_dim": input_dim,
            "hidden_layers": hidden_layers or [128, 64, 32],
            "dropout_rates": dropout_rates or [0.5, 0.3, 0.3],
            "use_batch_norm": use_batch_norm,
        }
        self.model = CreditNetReg(
            input_dim=input_dim,
            hidden_layers=self.arch_config["hidden_layers"],
            dropout_rates=self.arch_config["dropout_rates"],
            use_batch_norm=self.arch_config["use_batch_norm"],
        ).to(self.device)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.threshold = 0.5
        self.recommended_threshold = 0.5

    @staticmethod
    def _best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.5) -> float:
        candidates = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = (y_prob >= threshold).astype(int)
            precision = __import__("sklearn.metrics", fromlist=["precision_score"]).precision_score(
                y_true, y_pred, zero_division=0
            )
            if precision < min_precision:
                continue
            score = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                y_true, y_pred, zero_division=0
            )
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        return best_threshold

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        val_size: float = 0.2,
        patience: int = 10,
    ) -> dict[str, list[float]]:
        X_subtrain, X_val, y_subtrain, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )

        X_subtrain_t = torch.tensor(X_subtrain, dtype=torch.float32)
        y_subtrain_t = torch.tensor(y_subtrain.reshape(-1, 1), dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

        train_set = TensorDataset(X_subtrain_t, y_subtrain_t)
        val_set = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        y_train_pos = float(y_train.sum())
        y_train_neg = float(len(y_train) - y_train_pos)
        pos_weight = torch.tensor([y_train_neg / max(y_train_pos, 1.0)], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Усиленная регуляризация: weight_decay 1e-3 вместо 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
        )

        best_val_loss = float("inf")
        best_state = None
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_set)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_set)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        val_prob = self.predict_proba(X_val)
        self.recommended_threshold = self._best_threshold_by_f1(y_val, val_prob)
        self.threshold = 0.5
        return self.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).squeeze(1)
        return probs.detach().cpu().numpy()

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.arch_config["input_dim"],
            "arch_config": self.arch_config,
            "history": self.history,
            "threshold": self.threshold,
            "recommended_threshold": self.recommended_threshold,
        }
        torch.save(checkpoint, str(path))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "CreditTrainerReg":
        checkpoint = torch.load(str(path), map_location="cpu")
        arch = checkpoint.get("arch_config", {})
        trainer = cls(
            input_dim=checkpoint["input_dim"],
            device=device,
            hidden_layers=arch.get("hidden_layers"),
            dropout_rates=arch.get("dropout_rates"),
            use_batch_norm=arch.get("use_batch_norm", True),
        )
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        trainer.threshold = float(checkpoint.get("threshold", 0.5))
        trainer.recommended_threshold = float(checkpoint.get("recommended_threshold", trainer.threshold))
        trainer.model.eval()
        return trainer


# ---------------------------------------------------------------------------
# 2. Sklearn-модели
# ---------------------------------------------------------------------------
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    dt.fit(X_train, y_train)
    return dt


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    sample_weight = np.where(y_train == 1, neg / max(pos, 1.0), 1.0)
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    gb.fit(X_train, y_train, sample_weight=sample_weight)
    return gb


# ---------------------------------------------------------------------------
# 3. Оценка
# ---------------------------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, name: str, is_mlp: bool = False) -> dict:
    if is_mlp:
        train_prob = model.predict_proba(X_train)
        test_prob = model.predict_proba(X_test)
    else:
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_binary_metrics(y_train, train_prob, threshold=0.5)
    test_metrics = evaluate_binary_metrics(y_test, test_prob, threshold=0.5)

    return {
        "name": name,
        "train": train_metrics,
        "test": test_metrics,
        "gap": {k: train_metrics[k] - test_metrics[k] for k in train_metrics},
        "train_prob": train_prob,
        "test_prob": test_prob,
    }


def plot_training_comparison(histories: dict[str, dict], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories.items():
        axes[0].plot(hist["train_loss"], label=f"{name} train")
        axes[1].plot(hist["val_loss"], label=f"{name} val")

    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[experiments] Графики обучения сохранены: {save_path}")


def plot_roc_comparison(results: list[dict], y_test: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = {
        "CreditNet Original": "#e74c3c",
        "CreditNet Regularized": "#3498db",
        "Random Forest": "#2ecc71",
        "Gradient Boosting": "#f39c12",
        "Decision Tree": "#9b59b6",
    }

    for res in results:
        name = res["name"]
        fpr, tpr, _ = roc_curve(y_test, res["test_prob"])
        auc = roc_auc_score(y_test, res["test_prob"])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})", color=colors.get(name, "#333"))

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайный")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Сравнительная ROC-кривая (эксперименты)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[experiments] ROC-кривая сохранена: {save_path}")


def print_results_table(results: list[dict]) -> str:
    lines = []
    lines.append(f"{'Модель':<25} {'Metric':<10} {'Train':<10} {'Test':<10} {'Gap':<10}")
    lines.append("-" * 70)

    for res in results:
        name = res["name"]
        for metric in ["roc_auc", "f1", "precision", "recall"]:
            t = res["train"][metric]
            v = res["test"][metric]
            g = res["gap"][metric]
            lines.append(f"{name:<25} {metric:<10} {t:<10.4f} {v:<10.4f} {g:<+10.4f}")
        lines.append("")

    table = "\n".join(lines)
    return table


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперименты с моделями кредитного скоринга")
    parser.add_argument("--dataset", type=str, default="german")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = get_processed_dir(project_root, args.dataset)
    plots_dir = project_root / "data" / "plots" / "experiments"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    print(f"[experiments] Датасет: {args.dataset} | Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[experiments] Class balance: {(y_train==1).mean():.1%} bad | {(y_train==0).mean():.1%} good\n")

    results: list[dict] = []
    histories: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # 1. CreditNet Original (baseline)
    # -----------------------------------------------------------------------
    print("[experiments] 1/5 CreditNet Original...")
    mlp_orig_path = processed_dir / "model.pt"
    if mlp_orig_path.exists():
        trainer_orig = CreditTrainer.load(mlp_orig_path, device="cpu")
    else:
        trainer_orig = CreditTrainer(input_dim=X_train.shape[1], device="cpu")
        trainer_orig.train(X_train, y_train, epochs=args.epochs, lr=args.lr)
        trainer_orig.save(mlp_orig_path)

    res_orig = evaluate_model(trainer_orig, X_train, y_train, X_test, y_test, "CreditNet Original", is_mlp=True)
    results.append(res_orig)
    histories["CreditNet Original"] = trainer_orig.history

    # -----------------------------------------------------------------------
    # 2. CreditNet Regularized
    # -----------------------------------------------------------------------
    print("[experiments] 2/5 CreditNet Regularized...")
    trainer_reg = CreditTrainerReg(input_dim=X_train.shape[1], device="cpu")
    trainer_reg.train(X_train, y_train, epochs=args.epochs, lr=args.lr, batch_size=32, patience=10)
    res_reg = evaluate_model(trainer_reg, X_train, y_train, X_test, y_test, "CreditNet Regularized", is_mlp=True)
    results.append(res_reg)
    histories["CreditNet Regularized"] = trainer_reg.history

    # -----------------------------------------------------------------------
    # 3. Random Forest
    # -----------------------------------------------------------------------
    print("[experiments] 3/5 Random Forest...")
    rf = train_random_forest(X_train, y_train)
    res_rf = evaluate_model(rf, X_train, y_train, X_test, y_test, "Random Forest")
    results.append(res_rf)

    # -----------------------------------------------------------------------
    # 4. Gradient Boosting
    # -----------------------------------------------------------------------
    print("[experiments] 4/5 Gradient Boosting...")
    gb = train_gradient_boosting(X_train, y_train)
    res_gb = evaluate_model(gb, X_train, y_train, X_test, y_test, "Gradient Boosting")
    results.append(res_gb)

    # -----------------------------------------------------------------------
    # 5. Decision Tree
    # -----------------------------------------------------------------------
    print("[experiments] 5/5 Decision Tree...")
    dt = train_decision_tree(X_train, y_train)
    res_dt = evaluate_model(dt, X_train, y_train, X_test, y_test, "Decision Tree")
    results.append(res_dt)

    # -----------------------------------------------------------------------
    # Выводы и сохранение
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("=" * 70)
    table = print_results_table(results)
    print(table)

    # Графики
    plot_training_comparison(histories, plots_dir / "training_comparison.png")
    plot_roc_comparison(results, y_test, plots_dir / "roc_experiments.png")

    # JSON-отчёт
    report = {
        "dataset": args.dataset,
        "models": [
            {
                "name": r["name"],
                "train_metrics": r["train"],
                "test_metrics": r["test"],
                "gap": {k: float(v) for k, v in r["gap"].items()},
            }
            for r in results
        ],
    }
    report_path = reports_dir / "experiments_results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[experiments] Отчёт сохранён: {report_path}")

    # Markdown таблица
    md_path = reports_dir / "experiments_table.md"
    md_lines = ["# Результаты экспериментов\n", "| Модель | ROC-AUC (test) | F1 (test) | Precision (test) | Recall (test) | Gap ROC-AUC |", "|--------|----------------|-----------|------------------|---------------|-------------|"]
    for r in results:
        m = r["test"]
        g = r["gap"]["roc_auc"]
        md_lines.append(
            f"| {r['name']} | {m['roc_auc']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {g:+.4f} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[experiments] Markdown таблица: {md_path}")

    # Анализ переобучения
    print("\n" + "=" * 70)
    print("АНАЛИЗ ПЕРЕОБУЧЕНИЯ (gap train - test)")
    print("=" * 70)
    for r in results:
        name = r["name"]
        gap_auc = r["gap"]["roc_auc"]
        gap_f1 = r["gap"]["f1"]
        status = "🔴 Сильное" if gap_auc > 0.05 else "🟡 Умеренное" if gap_auc > 0.02 else "🟢 Низкое"
        print(f"{name:<25} ROC-AUC gap={gap_auc:+.4f}  F1 gap={gap_f1:+.4f}  {status}")

    # Лучшая модель
    best_by_auc = max(results, key=lambda r: r["test"]["roc_auc"])
    best_by_f1 = max(results, key=lambda r: r["test"]["f1"])
    print(f"\n🏆 Лучшая по ROC-AUC: {best_by_auc['name']} ({best_by_auc['test']['roc_auc']:.4f})")
    print(f"🏆 Лучшая по F1:      {best_by_f1['name']} ({best_by_f1['test']['f1']:.4f})")


if __name__ == "__main__":
    main()
