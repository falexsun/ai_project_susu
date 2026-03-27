from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch


@dataclass
class ExplainResult:
    shap_values: np.ndarray
    base_value: float
    feature_names: list[str]
    prediction: float


class CreditExplainer:
    def __init__(self, model: torch.nn.Module, X_train: np.ndarray, feature_names: list[str]) -> None:
        self.model = model.eval()
        self.feature_names = feature_names
        self.background = X_train[: min(200, len(X_train))]
        self._framework = "deep"

        try:
            background_t = torch.tensor(self.background, dtype=torch.float32)
            self.explainer = shap.DeepExplainer(self.model, background_t)
        except Exception:
            self._framework = "kernel"

            def predict_fn(x: np.ndarray) -> np.ndarray:
                xt = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    logits = self.model(xt)
                    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                return probs

            self.explainer = shap.KernelExplainer(predict_fn, self.background)

    def explain(self, X_sample: np.ndarray) -> dict[str, Any]:
        X_sample = np.asarray(X_sample, dtype=np.float32)
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        with torch.no_grad():
            pred = torch.sigmoid(
                self.model(torch.tensor(X_sample, dtype=torch.float32))
            ).cpu().numpy().reshape(-1)

        if self._framework == "deep":
            shap_values_raw = self.explainer.shap_values(torch.tensor(X_sample, dtype=torch.float32))
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[0]
            shap_values = np.array(shap_values_raw).reshape(X_sample.shape)
            expected = self.explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                base_value = float(np.array(expected).reshape(-1)[0])
            else:
                base_value = float(expected)
        else:
            shap_values_raw = self.explainer.shap_values(X_sample)
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[0]
            shap_values = np.array(shap_values_raw).reshape(X_sample.shape)
            expected = self.explainer.expected_value
            base_value = float(np.array(expected).reshape(-1)[0])

        return {
            "shap_values": shap_values[0],
            "base_value": base_value,
            "feature_names": self.feature_names,
            "prediction": float(pred[0]),
            "sample": X_sample[0],
        }

    def plot_waterfall(self, shap_values: np.ndarray, feature_names: list[str], prediction: float, sample: np.ndarray | None = None) -> None:
        shap_values = np.asarray(shap_values)
        sample = np.zeros_like(shap_values) if sample is None else np.asarray(sample)

        try:
            exp = shap.Explanation(
                values=shap_values,
                base_values=0.0,
                data=sample,
                feature_names=feature_names,
            )
            shap.plots.waterfall(exp, max_display=15, show=False)
            plt.title(f"Локальное объяснение SHAP, p(default)={prediction:.3f}")
            plt.tight_layout()
        except Exception:
            idx = np.argsort(np.abs(shap_values))[-15:]
            vals = shap_values[idx]
            names = [feature_names[i] for i in idx]
            colors = ["#c0392b" if v > 0 else "#27ae60" for v in vals]

            plt.figure(figsize=(9, 6))
            plt.barh(names, vals, color=colors)
            plt.axvline(0.0, color="black", linewidth=1)
            plt.title(f"Waterfall (fallback), p(default)={prediction:.3f}")
            plt.xlabel("Вклад SHAP")
            plt.tight_layout()

    def plot_global_importance(self, X_test: np.ndarray) -> None:
        X_test = np.asarray(X_test, dtype=np.float32)
        X_batch = X_test[: min(300, len(X_test))]

        if self._framework == "deep":
            shap_values_raw = self.explainer.shap_values(torch.tensor(X_batch, dtype=torch.float32))
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[0]
            shap_values = np.array(shap_values_raw)
        else:
            shap_values_raw = self.explainer.shap_values(X_batch)
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[0]
            shap_values = np.array(shap_values_raw)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order = np.argsort(mean_abs)[-15:]

        plt.figure(figsize=(10, 6))
        plt.barh([self.feature_names[i] for i in order], mean_abs[order], color="#2c7fb8")
        plt.title("Глобальная важность признаков (среднее |SHAP|)")
        plt.xlabel("Среднее абсолютное значение SHAP")
        plt.tight_layout()
