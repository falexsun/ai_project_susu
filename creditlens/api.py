from __future__ import annotations

from pathlib import Path
from typing import Any
import sys
import __main__

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import CreditTrainer, get_processed_dir
from src.preprocess import DATASET_CONFIGS, CreditPreprocessor, DatasetConfig


setattr(__main__, "CreditPreprocessor", CreditPreprocessor)
sys.modules["__main__"].CreditPreprocessor = CreditPreprocessor
setattr(__main__, "DatasetConfig", DatasetConfig)
sys.modules["__main__"].DatasetConfig = DatasetConfig


class PredictRequest(BaseModel):
    dataset: str = Field(default="german", description="Имя датасета/схемы")
    client_data: dict[str, Any] = Field(description="Признаки клиента в формате key:value")


app = FastAPI(
    title="CreditLens API",
    description="API расчета вероятности дефолта клиента на основе ensemble-модели",
    version="1.0.0",
)


def _load_artifacts(dataset: str) -> tuple[Any, Any, CreditTrainer]:
    if dataset not in DATASET_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Неизвестный dataset '{dataset}'.")

    project_root = Path(__file__).resolve().parent
    processed_dir = get_processed_dir(project_root, dataset)

    preprocessor_path = processed_dir / "preprocessor.pkl"
    ensemble_path = processed_dir / "ensemble.pkl"
    mlp_path = processed_dir / "model.pt"

    if not preprocessor_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Не найден preprocessor: {preprocessor_path}. "
                f"Сначала выполните preprocess для dataset={dataset}."
            ),
        )
    if not mlp_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Не найдена MLP модель: {mlp_path}. Сначала выполните обучение model.py.",
        )

    preprocessor = joblib.load(preprocessor_path)
    trainer = CreditTrainer.load(mlp_path, device="cpu")
    ensemble_artifact = joblib.load(ensemble_path) if ensemble_path.exists() else None
    return preprocessor, ensemble_artifact, trainer


def _predict_proba(X: np.ndarray, ensemble_artifact: Any, trainer: CreditTrainer) -> tuple[np.ndarray, str]:
    if ensemble_artifact is None:
        return trainer.predict_proba(X), "mlp"

    logreg = ensemble_artifact["logreg"]
    gb = ensemble_artifact["gb"]
    meta = ensemble_artifact["meta"]

    p_logreg = logreg.predict_proba(X)[:, 1]
    p_gb = gb.predict_proba(X)[:, 1]
    p_mlp = trainer.predict_proba(X)
    stacked = np.column_stack([p_logreg, p_gb, p_mlp])
    p_ens = meta.predict_proba(stacked)[:, 1]
    return p_ens, "ensemble"


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "CreditLens API"}


@app.get("/datasets")
def list_datasets() -> dict[str, list[str]]:
    return {"datasets": sorted(DATASET_CONFIGS.keys())}


@app.get("/schema/{dataset}")
def get_schema(dataset: str) -> dict[str, Any]:
    preprocessor, _, _ = _load_artifacts(dataset)
    feature_columns = list(getattr(preprocessor, "feature_columns", []))

    return {
        "dataset": dataset,
        "required_client_fields": feature_columns,
        "fields_count": len(feature_columns),
        "note": "Передавайте JSON с полями из required_client_fields.",
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    preprocessor, ensemble_artifact, trainer = _load_artifacts(request.dataset)

    feature_columns = list(getattr(preprocessor, "feature_columns", []))
    missing = [name for name in feature_columns if name not in request.client_data]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "В запросе отсутствуют обязательные поля клиента.",
                "missing_fields": missing,
            },
        )

    client_df = pd.DataFrame([request.client_data])
    try:
        X = preprocessor.transform(client_df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Ошибка преобразования признаков: {exc}") from exc

    prob, model_type = _predict_proba(X, ensemble_artifact, trainer)
    probability_default = float(prob[0])

    threshold = 0.5
    if ensemble_artifact is not None:
        threshold = float(ensemble_artifact.get("threshold", 0.5))
    else:
        threshold = float(getattr(trainer, "threshold", 0.5))

    decision = "REJECT" if probability_default >= threshold else "APPROVE"

    return {
        "dataset": request.dataset,
        "model_type": model_type,
        "probability_default": probability_default,
        "threshold": threshold,
        "decision": decision,
    }
