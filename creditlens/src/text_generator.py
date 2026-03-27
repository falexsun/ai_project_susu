from __future__ import annotations

from typing import Any

import numpy as np


FEATURE_TEMPLATES = {
    "Amount": "Сумма кредита",
    "Duration": "Срок кредита",
    "Age": "Возраст заемщика",
    "History": "Кредитная история",
    "Employment": "Срок занятости",
    "Status": "Статус счета",
    "Purpose": "Цель кредита",
    "Housing": "Тип жилья",
    "Job": "Тип занятости",
    "InstallmentRate": "Нагрузка по платежу",
    "PersonalStatus": "Семейный и социальный статус",
}


def _normalize_feature_name(feature_name: str) -> str:
    raw = feature_name.split("__")[-1]
    base = raw.split("_")[0]
    return FEATURE_TEMPLATES.get(base, raw)


def generate_explanation(shap_dict: dict[str, Any]) -> str:
    shap_values = np.asarray(shap_dict["shap_values"])
    feature_names = list(shap_dict["feature_names"])
    prediction = float(shap_dict["prediction"])

    decision = "ОТКАЗ" if prediction >= 0.5 else "ОДОБРЕНИЕ"

    pos_idx = np.argsort(shap_values)[-3:][::-1]
    neg_idx = np.argsort(shap_values)[:2]

    risk_factors = []
    for i in pos_idx:
        if shap_values[i] > 0:
            risk_factors.append(f"{_normalize_feature_name(feature_names[i])} (+{shap_values[i]:.3f})")

    protective_factors = []
    for i in neg_idx:
        if shap_values[i] < 0:
            protective_factors.append(f"{_normalize_feature_name(feature_names[i])} ({shap_values[i]:.3f})")

    risk_text = ", ".join(risk_factors) if risk_factors else "выраженные риск-факторы не обнаружены"
    prot_text = ", ".join(protective_factors) if protective_factors else "защитные факторы выражены слабо"

    return (
        f"Решение: {decision}. "
        f"Вероятность дефолта: {prediction * 100:.1f}%. "
        f"Основные причины риска: {risk_text}. "
        f"Факторы, снижающие риск: {prot_text}."
    )
