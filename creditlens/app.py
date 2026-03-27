from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.explainer import CreditExplainer
from src.model import CreditTrainer
from src.text_generator import generate_explanation


st.set_page_config(page_title="CreditLens", page_icon="📊", layout="wide")
st.title("CreditLens — Интерпретируемый кредитный скоринг")


@st.cache_resource
def load_artifacts() -> tuple[CreditTrainer, object, list[str], np.ndarray, np.ndarray]:
    root = Path(__file__).resolve().parent
    processed = root / "data" / "processed"

    trainer = CreditTrainer.load(processed / "model.pt", device="cpu")
    preprocessor = joblib.load(processed / "preprocessor.pkl")
    feature_names = json.loads((processed / "feature_names.json").read_text(encoding="utf-8"))
    X_train = np.load(processed / "X_train.npy")
    X_test = np.load(processed / "X_test.npy")
    return trainer, preprocessor, feature_names, X_train, X_test


def build_input_df(
    age: int,
    amount: int,
    duration: int,
    history: str,
    purpose: str,
    employment: str,
) -> pd.DataFrame:
    # Поля, используемые в пайплайне, заполняются значениями по умолчанию, если не введены пользователем.
    return pd.DataFrame(
        [
            {
                "Status": "A14",
                "History": history,
                "Purpose": purpose,
                "PersonalStatus": "A93",
                "Housing": "A152",
                "Job": employment,
                "Amount": amount,
                "Duration": duration,
                "Age": age,
                "InstallmentRate": 2,
            }
        ]
    )


trainer, preprocessor, feature_names, X_train, X_test = load_artifacts()

history_map = {
    "нет кредитов": "A30",
    "все погашены": "A31",
    "существующие погашены": "A32",
    "задержки": "A33",
    "критический счет": "A34",
}
purpose_map = {
    "авто": "A41",
    "мебель": "A42",
    "электроника": "A43",
    "образование": "A47",
    "бизнес": "A49",
    "другое": "A410",
}
employment_map = {
    "безработный": "A171",
    "до 1 года": "A172",
    "1–4 года": "A173",
    "4–7 лет": "A174",
    "более 7 лет": "A174",
}

with st.sidebar:
    st.header("Параметры клиента")
    age = st.slider("Возраст", 18, 75, 35)
    amount = st.slider("Сумма кредита", 500, 20000, 5000, step=100)
    duration = st.slider("Срок кредита (мес)", 6, 72, 24)

    history_rus = st.selectbox("Кредитная история", list(history_map.keys()))
    purpose_rus = st.selectbox("Цель кредита", list(purpose_map.keys()))
    employment_rus = st.selectbox("Занятость", list(employment_map.keys()))

    run_button = st.button("Оценить заявку", use_container_width=True)

if run_button:
    row = build_input_df(
        age=age,
        amount=amount,
        duration=duration,
        history=history_map[history_rus],
        purpose=purpose_map[purpose_rus],
        employment=employment_map[employment_rus],
    )

    X_user = preprocessor.transform(row)
    pred = float(trainer.predict_proba(X_user)[0])
    threshold = float(getattr(trainer, "threshold", 0.5))

    approved = pred < threshold
    decision = "ОДОБРЕНО" if approved else "ОТКАЗ"
    color = "#1f9d55" if approved else "#d64545"

    st.markdown(
        f"""
        <div style='padding:16px;border-radius:14px;background:{color};color:white;text-align:center;'>
            <h2 style='margin:0;'>{decision}</h2>
            <p style='margin:4px 0 0 0;font-size:20px;'>Вероятность дефолта: {pred*100:.1f}%</p>
            <p style='margin:4px 0 0 0;font-size:16px;'>Порог решения: {threshold*100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    explainer = CreditExplainer(trainer.model, X_train, feature_names)
    shap_info = explainer.explain(X_user)

    st.subheader("Текстовое объяснение")
    st.write(generate_explanation(shap_info))

    st.subheader("SHAP waterfall")
    fig = plt.figure(figsize=(9, 6))
    explainer.plot_waterfall(
        shap_values=shap_info["shap_values"],
        feature_names=shap_info["feature_names"],
        prediction=shap_info["prediction"],
        sample=shap_info["sample"],
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("What-if анализ")
    amount_whatif = st.slider(
        "Изменить сумму кредита", 500, 20000, int(amount), step=100, key="whatif_amount"
    )
    duration_whatif = st.slider(
        "Изменить срок кредита", 6, 72, int(duration), key="whatif_duration"
    )

    row_whatif = build_input_df(
        age=age,
        amount=amount_whatif,
        duration=duration_whatif,
        history=history_map[history_rus],
        purpose=purpose_map[purpose_rus],
        employment=employment_map[employment_rus],
    )
    X_whatif = preprocessor.transform(row_whatif)
    pred_whatif = float(trainer.predict_proba(X_whatif)[0])

    delta = pred_whatif - pred
    st.write(
        f"Новая вероятность дефолта: **{pred_whatif*100:.1f}%** "
        f"(изменение {delta*100:+.1f} п.п.)"
    )
else:
    st.info("Заполните параметры слева и нажмите «Оценить заявку».")
