"""
Unit-тесты для модуля model.py (CreditNet / CreditTrainer).

Запуск:
    cd creditlens && python -m pytest tests/test_model.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import CreditNet, CreditTrainer, evaluate_binary_metrics


class TestCreditNet:
    def test_forward_shape(self) -> None:
        batch_size, input_dim = 16, 20
        model = CreditNet(input_dim)
        x = torch.randn(batch_size, input_dim)
        out = model(x)
        assert out.shape == (batch_size, 1)

    def test_output_range_sigmoid(self) -> None:
        model = CreditNet(10)
        x = torch.randn(32, 10)
        logits = model(x)
        probs = torch.sigmoid(logits)
        assert torch.all((probs >= 0) & (probs <= 1))


class TestCreditTrainer:
    def test_train_shapes(self) -> None:
        trainer = CreditTrainer(input_dim=5, device="cpu")
        X = np.random.randn(200, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=200).astype(np.int64)
        history = trainer.train(X, y, epochs=3, batch_size=32)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == len(history["val_loss"])
        assert len(history["train_loss"]) <= 3

    def test_predict_proba(self) -> None:
        trainer = CreditTrainer(input_dim=5, device="cpu")
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=50).astype(np.int64)
        trainer.train(X, y, epochs=2, batch_size=16)
        probs = trainer.predict_proba(X)
        assert probs.shape == (50,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_save_load(self) -> None:
        trainer = CreditTrainer(input_dim=4, device="cpu")
        X = np.random.randn(100, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=100).astype(np.int64)
        trainer.train(X, y, epochs=2, batch_size=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)
            loaded = CreditTrainer.load(path, device="cpu")

        assert loaded.model.layers[0].in_features == 4
        probs_before = trainer.predict_proba(X)
        probs_after = loaded.predict_proba(X)
        np.testing.assert_allclose(probs_before, probs_after, atol=1e-6)

    def test_threshold_optimization(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        best = CreditTrainer._best_threshold_by_f1(y_true, y_prob, min_precision=0.5)
        assert 0.1 <= best <= 0.9


class TestEvaluateBinaryMetrics:
    def test_perfect_classifier(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        metrics = evaluate_binary_metrics(y_true, y_prob)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_random_classifier(self) -> None:
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=200)
        y_prob = np.random.rand(200)
        metrics = evaluate_binary_metrics(y_true, y_prob)
        assert 0.4 <= metrics["roc_auc"] <= 0.6  # примерно случайно


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
