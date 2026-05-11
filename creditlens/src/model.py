from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from training_utils import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
    save_metrics_report,
)


def get_processed_dir(project_root: Path, dataset: str) -> Path:
    if dataset == "german":
        return project_root / "data" / "processed"
    return project_root / "data" / "processed" / dataset


class CreditNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | None = None,
        dropout_rates: list[float] | None = None,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [256, 128, 64]
        dropout_rates = dropout_rates or [0.3, 0.2, 0.2]

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


class CreditTrainer:
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
            "hidden_layers": hidden_layers or [256, 128, 64],
            "dropout_rates": dropout_rates or [0.3, 0.2, 0.2],
            "use_batch_norm": use_batch_norm,
        }
        self.model = CreditNet(
            input_dim=input_dim,
            hidden_layers=self.arch_config["hidden_layers"],
            dropout_rates=self.arch_config["dropout_rates"],
            use_batch_norm=self.arch_config["use_batch_norm"],
        ).to(self.device)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.threshold = 0.5
        self.recommended_threshold = 0.5

    @staticmethod
    def _best_threshold_by_f1(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        min_precision: float = 0.5,
    ) -> float:
        candidates = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            if precision < min_precision:
                continue
            score = f1_score(y_true, y_pred, zero_division=0)
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
        batch_size: int = 64,
        val_size: float = 0.2,
        patience: int = 15,
    ) -> dict[str, list[float]]:
        X_subtrain, X_val, y_subtrain, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=42,
            stratify=y_train,
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
        pos_weight = torch.tensor([
            y_train_neg / max(y_train_pos, 1.0)
        ], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
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

            if epoch % 10 == 0 or epoch == 1:
                print(f"Эпоха {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(f"Early stopping на эпохе {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        val_prob = self.predict_proba(X_val)
        self.recommended_threshold = self._best_threshold_by_f1(y_val, val_prob)
        self.threshold = 0.5
        print(
            "Пороги: "
            f"prod={self.threshold:.3f}, "
            f"recommended(val)={self.recommended_threshold:.3f}"
        )

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
    def load(cls, path: str | Path, device: str | None = None) -> "CreditTrainer":
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


def evaluate_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение модели кредитного скоринга")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "uci_credit_card", "give_me_some_credit", "home_credit"],
        help="Название датасета",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=0.001, help="Скорость обучения")
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

    print(f"[info] Датасет: {args.dataset} | X_train: {X_train.shape} | X_test: {X_test.shape}")

    trainer = CreditTrainer(input_dim=X_train.shape[1])

    started = time.perf_counter()
    history = trainer.train(X_train, y_train, epochs=args.epochs, lr=args.lr)
    duration = time.perf_counter() - started

    test_prob = trainer.predict_proba(X_test)
    metrics = evaluate_binary_metrics(y_test, test_prob, threshold=trainer.threshold)

    # Сохранение модели
    model_path = processed_dir / "model.pt"
    trainer.save(model_path)

    # Визуализация кривых обучения
    plot_training_curves(
        history,
        save_path=plots_dir / "training_curves.png",
        title=f"Кривые обучения — {args.dataset}",
    )

    # ROC и PR кривые
    plot_roc_curve(
        y_test, test_prob,
        save_path=plots_dir / "roc_curve.png",
        model_name="CreditNet (MLP)",
    )
    plot_pr_curve(
        y_test, test_prob,
        save_path=plots_dir / "pr_curve.png",
        model_name="CreditNet (MLP)",
    )

    # Confusion matrix
    y_pred = (test_prob >= trainer.threshold).astype(int)
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=plots_dir / "confusion_matrix.png",
        labels=["Одобрено (0)", "Дефолт (1)"],
    )

    # Сохранение метрик в JSON
    save_metrics_report(
        metrics,
        save_path=plots_dir / "metrics.json",
        extra={
            "dataset": args.dataset,
            "model": "CreditNet",
            "epochs": len(history["train_loss"]),
            "lr": args.lr,
            "threshold": trainer.threshold,
            "recommended_threshold": trainer.recommended_threshold,
            "training_time_sec": round(duration, 2),
        },
    )

    print(f"\n{'='*50}")
    print(f"Датасет: {args.dataset}")
    print(f"Модель сохранена: {model_path}")
    print(f"Время обучения: {duration:.2f} сек")
    print("Метрики на тесте:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  threshold: {trainer.threshold:.3f}")
    print(f"  recommended_threshold: {trainer.recommended_threshold:.3f}")
    print(f"Графики сохранены в: {plots_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
