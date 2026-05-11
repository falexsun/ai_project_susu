from __future__ import annotations

import json
import sys
import __main__
from pathlib import Path
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.explainer import CreditExplainer
from src.model import CreditTrainer
from src.preprocess import CreditPreprocessor, DatasetConfig
from src.text_generator import (
    explain_waterfall_for_user,
    generate_explanation,
    generate_human_brief,
    humanize_feature_name,
    generate_segment_risk,
    generate_auto_recommendations,
)

# ---------------------------------------------------------------------------
# monkey-patch so joblib can find the classes
# ---------------------------------------------------------------------------
setattr(__main__, "CreditPreprocessor", CreditPreprocessor)
sys.modules["__main__"].CreditPreprocessor = CreditPreprocessor
setattr(__main__, "DatasetConfig", DatasetConfig)
sys.modules["__main__"].DatasetConfig = DatasetConfig

DM_TO_RUB = 50

st.set_page_config(
    page_title="CreditLens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# inject custom css — quick'n'dirty way, should be refactored
_css_path = Path(__file__).resolve().parent / "assets" / "custom.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="cl-header">
        <h1>CreditLens</h1>
        <p>Скоринговая система с интерпретацией решений. Заполните профиль клиента, чтобы получить оценку риска и рекомендации.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "show_results" not in st.session_state:
    st.session_state.show_results = False


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
    status: str,
    history: str,
    purpose: str,
    savings: str,
    employment_tenure: str,
    installment_rate: int,
    personal_status: str,
    guarantors: str,
    residence: int,
    property_type: str,
    other_installments: str,
    housing: str,
    existing_credits: int,
    job: str,
    dependents: int,
    phone: str,
    foreign: str,
) -> pd.DataFrame:
    # полный набор полей german-датасета для preprocessor.transform
    defaults = {
        "Status": status,
        "Duration": duration,
        "History": history,
        "Purpose": purpose,
        "Amount": amount,
        "Savings": savings,
        "Employment": employment_tenure,
        "InstallmentRate": installment_rate,
        "PersonalStatus": personal_status,
        "Guarantors": guarantors,
        "Residence": residence,
        "Property": property_type,
        "Age": age,
        "OtherInstallments": other_installments,
        "Housing": housing,
        "ExistingCredits": existing_credits,
        "Job": job,
        "Dependents": dependents,
        "Phone": phone,
        "Foreign": foreign,
    }
    return pd.DataFrame([defaults])


trainer, preprocessor, feature_names, X_train, X_test = load_artifacts()

# ---------------------------------------------------------------------------
# Mappings — ручной перевод категорий датасета в человекочитаемые labels
# ---------------------------------------------------------------------------
history_map = {
    "ранее кредиты погашались вовремя": "A30",
    "кредиты в этом банке закрыты": "A31",
    "текущие кредиты без просрочек": "A32",
    "были задержки платежей": "A33",
    "критичная история / проблемные кредиты": "A34",
}
purpose_map = {
    "автомобиль (новый)": "A40",
    "автомобиль (с пробегом)": "A41",
    "мебель или оборудование": "A42",
    "радио/ТВ": "A43",
    "домашние приборы": "A44",
    "ремонт": "A45",
    "образование": "A46",
    "отпуск": "A47",
    "переобучение": "A48",
    "бизнес": "A49",
    "прочие цели": "A410",
}
status_map = {
    "нет расчетного счета": "A14",
    "отрицательный остаток (<0 ₽)": "A11",
    "от 0 до 10 000 ₽": "A12",
    "свыше 10 000 ₽ / зарплатный счет": "A13",
}
savings_map = {
    "сбережений нет или неизвестно": "A65",
    "менее 5 000 ₽": "A61",
    "5 000 – 25 000 ₽": "A62",
    "25 000 – 50 000 ₽": "A63",
    "более 50 000 ₽": "A64",
}
employment_tenure_map = {
    "безработный": "A71",
    "стаж < 1 года": "A72",
    "стаж 1-4 года": "A73",
    "стаж 4-7 лет": "A74",
    "стаж > 7 лет": "A75",
}
job_map = {
    "безработный/нерезидент": "A171",
    "неквалифицированный": "A172",
    "квалифицированный": "A173",
    "руководитель / высококвалифицированный": "A174",
}
personal_status_map = {
    "мужчина, разведен/отдельно": "A91",
    "женщина, замужем/разведена": "A92",
    "мужчина, не женат": "A93",
    "мужчина, женат/вдовец": "A94",
    "женщина, не замужем": "A95",
}
installment_rate_map = {
    ">= 35% от дохода (4)": 4,
    "25% - 35% от дохода (3)": 3,
    "20% - 25% от дохода (2)": 2,
    "< 20% от дохода (1)": 1,
}
residence_map = {
    "менее 1 года (1)": 1,
    "от 1 до 4 лет (2)": 2,
    "от 4 до 7 лет (3)": 3,
    "более 7 лет (4)": 4,
}
guarantors_map = {
    "нет": "A101",
    "созаемщик": "A102",
    "поручитель": "A103",
}
property_map = {
    "недвижимость": "A121",
    "страхование/накопления": "A122",
    "авто/прочее имущество": "A123",
    "имущество не подтверждено": "A124",
}
other_installments_map = {
    "нет": "A143",
    "банк": "A141",
    "магазин": "A142",
}
housing_map = {
    "собственное": "A152",
    "аренда": "A151",
    "бесплатное проживание": "A153",
}
dependents_map = {
    "от 0 до 2": 1,
    "3 и более": 2,
}
phone_map = {
    "нет подтвержденного телефона": "A191",
    "телефон подтвержден": "A192",
}
foreign_map = {
    "да": "A201",
    "нет": "A202",
}

# ---------------------------------------------------------------------------
# Sidebar — форма
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Профиль клиента")

    st.markdown("<p class='cl-muted' style='margin-bottom:12px;'>Основные параметры</p>", unsafe_allow_html=True)
    age = st.slider("Возраст", 18, 75, 35)
    amount_rub = st.slider("Сумма кредита, ₽", 25000, 1000000, 250000, step=5000)
    duration = st.slider("Срок, месяцев", 6, 72, 24)

    status_rus = st.selectbox("Статус расчетного счета", list(status_map.keys()))
    history_rus = st.selectbox("Кредитная история", list(history_map.keys()))
    purpose_rus = st.selectbox("Цель кредита", list(purpose_map.keys()))
    tenure_rus = st.selectbox("Стаж занятости", list(employment_tenure_map.keys()))

    with st.expander("Расширенные параметры", expanded=False):
        savings_rus = st.selectbox("Сбережения", list(savings_map.keys()))
        job_rus = st.selectbox("Квалификация работы", list(job_map.keys()))
        installment_rate_rus = st.selectbox("Платежная нагрузка", list(installment_rate_map.keys()))
        installment_rate = installment_rate_map[installment_rate_rus]
        personal_status_rus = st.selectbox("Семейный статус", list(personal_status_map.keys()))
        guarantors_rus = st.selectbox("Поручители / созаемщики", list(guarantors_map.keys()))
        residence_rus = st.selectbox("Срок проживания", list(residence_map.keys()))
        residence = residence_map[residence_rus]
        property_rus = st.selectbox("Тип имущества", list(property_map.keys()))
        other_installments_rus = st.selectbox("Другие рассрочки", list(other_installments_map.keys()))
        housing_rus = st.selectbox("Тип жилья", list(housing_map.keys()))
        existing_credits = st.slider("Действующие кредиты", 1, 4, 1)
        dependents_rus = st.selectbox("Иждивенцы", list(dependents_map.keys()))
        dependents = dependents_map[dependents_rus]
        phone_rus = st.selectbox("Телефон подтвержден", list(phone_map.keys()))
        foreign_rus = st.selectbox("Иностранный заемщик", list(foreign_map.keys()))

    st.markdown("<br/>", unsafe_allow_html=True)
    run_button = st.button("Оценить", type="primary", use_container_width=True)

if run_button:
    st.session_state.show_results = True

# ---------------------------------------------------------------------------
# Main content — результаты
# ---------------------------------------------------------------------------
if st.session_state.show_results:
    amount_dm = int(amount_rub / DM_TO_RUB)
    row = build_input_df(
        age=age,
        amount=amount_dm,
        duration=duration,
        status=status_map[status_rus],
        history=history_map[history_rus],
        purpose=purpose_map[purpose_rus],
        savings=savings_map[savings_rus],
        employment_tenure=employment_tenure_map[tenure_rus],
        installment_rate=installment_rate,
        personal_status=personal_status_map[personal_status_rus],
        guarantors=guarantors_map[guarantors_rus],
        residence=residence,
        property_type=property_map[property_rus],
        other_installments=other_installments_map[other_installments_rus],
        housing=housing_map[housing_rus],
        existing_credits=existing_credits,
        job=job_map[job_rus],
        dependents=dependents,
        phone=phone_map[phone_rus],
        foreign=foreign_map[foreign_rus],
    )

    X_user = preprocessor.transform(row)
    pred = float(trainer.predict_proba(X_user)[0])
    threshold = float(getattr(trainer, "threshold", 0.5))

    approved = pred < threshold
    decision = "Одобрено" if approved else "Отказ"
    decision_color = "#16a34a" if approved else "#dc2626"

    # ---- metrics row ----
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class='cl-metric'>
                <div class='cl-metric-label'>Решение</div>
                <div class='cl-metric-value' style='color:{decision_color};'>{decision}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        risk_color = "#dc2626" if pred >= threshold else "#16a34a" if pred < threshold * 0.7 else "#d97706"
        st.markdown(
            f"""<div class='cl-metric'>
                <div class='cl-metric-label'>Вероятность дефолта</div>
                <div class='cl-metric-value' style='color:{risk_color};'>{pred*100:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class='cl-metric'>
                <div class='cl-metric-label'>Порог модели</div>
                <div class='cl-metric-value' style='color:#334155;'>{threshold*100:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # risk bar — simple visual indicator
    bar_pct = min(int(pred * 100), 100)
    bar_color = "#16a34a" if bar_pct < 30 else "#d97706" if bar_pct < 70 else "#dc2626"
    st.markdown(
        f"""<div style="margin-top:4px;">
            <div style="height:4px; background:#e2e8f0; border-radius:2px; overflow:hidden;">
                <div style="width:{bar_pct}%; height:100%; background:{bar_color}; border-radius:2px; transition:width 0.3s;"></div>
            </div>
            <p class='cl-small' style='margin-top:4px;'>Интенсивность риска</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Interpretation + recommendations
    # -----------------------------------------------------------------------
    explainer = CreditExplainer(trainer.model, X_train, feature_names)
    shap_info = explainer.explain(X_user)

    st.markdown("<div class='cl-section'>Обоснование решения</div>", unsafe_allow_html=True)
    st.write(generate_explanation(shap_info))

    brief = generate_human_brief(shap_info)
    st.markdown(
        f"<p class='cl-muted'>Уровень риска: <b>{brief['risk_level'].capitalize()}</b> &middot; "
        f"Вероятность затруднений: <b>{brief['probability']*100:.1f}%</b></p>",
        unsafe_allow_html=True,
    )

    if brief["reasons"]:
        st.markdown("<p class='cl-muted'>Ключевые факторы:</p>", unsafe_allow_html=True)
        for reason in brief["reasons"]:
            st.write(f"– {reason}")

    auto_recs = generate_auto_recommendations(shap_info, amount_rub, duration)
    if auto_recs:
        st.markdown("<div class='cl-section'>Рекомендации</div>", unsafe_allow_html=True)
        for rec in auto_recs:
            st.markdown(f"<div class='cl-rec'>{rec}</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # What-if (compact inline)
    # -----------------------------------------------------------------------
    st.markdown("<div class='cl-section'>What-if анализ</div>", unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        amount_whatif_rub = st.slider(
            "Сумма кредита", 25000, 1000000, int(amount_rub), step=5000, key="whatif_amount"
        )
    with w2:
        duration_whatif = st.slider(
            "Срок кредита", 6, 72, int(duration), key="whatif_duration"
        )

    amount_whatif_dm = int(amount_whatif_rub / DM_TO_RUB)
    row_whatif = build_input_df(
        age=age,
        amount=amount_whatif_dm,
        duration=duration_whatif,
        status=status_map[status_rus],
        history=history_map[history_rus],
        purpose=purpose_map[purpose_rus],
        savings=savings_map[savings_rus],
        employment_tenure=employment_tenure_map[tenure_rus],
        installment_rate=installment_rate,
        personal_status=personal_status_map[personal_status_rus],
        guarantors=guarantors_map[guarantors_rus],
        residence=residence,
        property_type=property_map[property_rus],
        other_installments=other_installments_map[other_installments_rus],
        housing=housing_map[housing_rus],
        existing_credits=existing_credits,
        job=job_map[job_rus],
        dependents=dependents,
        phone=phone_map[phone_rus],
        foreign=foreign_map[foreign_rus],
    )
    X_whatif = preprocessor.transform(row_whatif)
    pred_whatif = float(trainer.predict_proba(X_whatif)[0])
    delta = pred_whatif - pred

    delta_color = "#16a34a" if delta < 0 else "#dc2626" if delta > 0 else "#64748b"
    st.markdown(
        f"<p>Новая вероятность дефолта: <b>{pred_whatif*100:.1f}%</b> "
        f"<span style='color:{delta_color}; font-weight:600;'>({delta*100:+.1f} п.п.)</span></p>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # SHAP factors (collapsible, for operators)
    # -----------------------------------------------------------------------
    with st.expander("Детальный разбор факторов", expanded=False):
        friendly_feature_names = [humanize_feature_name(name) for name in shap_info["feature_names"]]
        top_idx = np.argsort(np.abs(shap_info["shap_values"]))[-8:][::-1]
        factors_df = pd.DataFrame(
            {
                "Фактор": [friendly_feature_names[i] for i in top_idx],
                "Влияние": [
                    "повышает" if shap_info["shap_values"][i] > 0 else "снижает"
                    for i in top_idx
                ],
                "Сила влияния": [float(abs(shap_info["shap_values"][i])) for i in top_idx],
            }
        )
        # color-coding via column config
        st.dataframe(
            factors_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Влияние": st.column_config.TextColumn(
                    "Влияние",
                    help="Направление влияния фактора на риск дефолта",
                ),
                "Сила влияния": st.column_config.NumberColumn(
                    "Сила",
                    format="%.4f",
                    help="Абсолютное значение вклада фактора",
                ),
            },
        )

        if st.checkbox("Показать SHAP waterfall", value=False):
            fig = plt.figure(figsize=(9, 6))
            explainer.plot_waterfall(
                shap_values=shap_info["shap_values"],
                feature_names=friendly_feature_names,
                prediction=shap_info["prediction"],
                sample=shap_info["sample"],
            )
            st.pyplot(fig)
            plt.close(fig)

    # -----------------------------------------------------------------------
    # Segment profile
    # -----------------------------------------------------------------------
    seg_risk = generate_segment_risk(age, job_rus)
    st.markdown("<div class='cl-section'>Сегментный профиль</div>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='cl-muted'>Группа: <b>{seg_risk['group']}</b> &middot; "
        f"Возрастной фактор: {seg_risk['age_risk']} &middot; "
        f"Профессиональный фактор: {seg_risk['job_risk']}</p>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Technical details (hidden by default)
    # -----------------------------------------------------------------------
    with st.expander("Техническая информация", expanded=False):
        ablation_path = Path(__file__).resolve().parent / "reports" / "ablation_summary.json"
        if ablation_path.exists():
            ablation_data = json.loads(ablation_path.read_text(encoding="utf-8"))
            experiments = ablation_data.get("experiments", {})
            rows = []
            for name, res in experiments.items():
                mean = res["mean"]
                std = res["std"]
                rows.append({
                    "Конфигурация": name,
                    "ROC-AUC": f"{mean['roc_auc']:.4f} ± {std['roc_auc']:.4f}",
                    "PR-AUC": f"{mean['pr_auc']:.4f} ± {std['pr_auc']:.4f}",
                    "F1": f"{mean['f1']:.4f} ± {std['f1']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            best = ablation_data.get("best_config", "—")
            test_m = ablation_data.get("test_metrics", {})
            st.markdown(
                f"<p class='cl-muted'>Лучшая конфигурация: <b>{best}</b> &middot; "
                f"ROC-AUC на тесте: <b>{test_m.get('roc_auc', 0):.4f}</b></p>",
                unsafe_allow_html=True,
            )
        else:
            st.write("Ablation study не проведен.")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    report_client_input = row.to_dict(orient="records")[0]
    report_client_input["Amount"] = amount_rub
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "probability_default": round(pred, 6),
        "decision": decision,
        "threshold": round(threshold, 6),
        "what_if_probability_default": round(pred_whatif, 6),
        "what_if_delta_pp": round(delta * 100, 3),
        "client_input": report_client_input,
    }
    st.download_button(
        "Скачать отчет (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="creditlens_report.json",
        mime="application/json",
    )

    # tiny footer
    st.markdown(
        "<p class='cl-small' style='margin-top:32px; text-align:center;'>CreditLens — внутренний инструмент скоринга</p>",
        unsafe_allow_html=True,
    )

else:
    st.info("Заполните параметры в боковой панели и нажмите «Оценить», чтобы получить результат.")
