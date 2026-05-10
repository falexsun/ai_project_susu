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


setattr(__main__, "CreditPreprocessor", CreditPreprocessor)
sys.modules["__main__"].CreditPreprocessor = CreditPreprocessor
setattr(__main__, "DatasetConfig", DatasetConfig)
sys.modules["__main__"].DatasetConfig = DatasetConfig


st.set_page_config(page_title="CreditLens", page_icon=None, layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }
    
    .hero {
        padding: 40px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(88,101,242,0.12) 0%, rgba(34,197,94,0.1) 100%);
        border: 1px solid rgba(128,128,128,0.15);
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .hero h1 { 
        margin: 0 0 10px 0; 
        color: var(--text-color); 
        font-weight: 800;
        font-size: 2.5rem !important;
    }
    .hero p { 
        margin: 0; 
        color: var(--text-color); 
        font-size: 1.1rem;
        opacity: 0.85;
    }
    
    .metric-card {
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.2);
        background: var(--secondary-background-color);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px -5px rgba(0,0,0,0.15);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    
    div[data-testid="stExpander"] details summary {
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }
    
    .info-panel {
        background: rgba(88,101,242,0.05);
        border-left: 4px solid #5865F2;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1> CreditLens</h1>
        <p>Мы помогаем с умом подходить к финансовым решениям. Наш алгоритм не просто выдает ответ, но и подробно, по-человечески всё объясняет. Заполните небольшую анкету, чтобы получить разбор вашей ситуации и полезные персональные советы.</p>
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
    # Для transform нужен полный набор полей german-датасета.
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
    "отрицательный остаток (<0 DM)": "A11",
    "от 0 до 200 DM": "A12",
    "свыше 200 DM / зарплатный счет": "A13",
}
savings_map = {
    "сбережений нет или неизвестно": "A65",
    "менее 100 DM": "A61",
    "100–500 DM": "A62",
    "500–1000 DM": "A63",
    "более 1000 DM": "A64",
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
    "≥ 35% от дохода (4)": 4,
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

with st.sidebar:
    st.header(" Анкета клиента")

    st.markdown("###  Основные параметры")
    age = st.slider("Возраст", 18, 75, 35)
    amount = st.slider("Сумма кредита", 500, 20000, 5000, step=100)
    duration = st.slider("Срок кредита (мес)", 6, 72, 24)

    status_rus = st.selectbox("Статус расчетного счета", list(status_map.keys()))
    history_rus = st.selectbox("Кредитная история", list(history_map.keys()))
    purpose_rus = st.selectbox("Цель кредита", list(purpose_map.keys()))
    tenure_rus = st.selectbox("Стаж занятости", list(employment_tenure_map.keys()))

    with st.expander(" Расширенные параметры", expanded=False):
        savings_rus = st.selectbox("Сбережения", list(savings_map.keys()))
        job_rus = st.selectbox("Квалификация работы", list(job_map.keys()))
        installment_rate_rus = st.selectbox("Платежная нагрузка", list(installment_rate_map.keys()))
        installment_rate = installment_rate_map[installment_rate_rus]
        personal_status_rus = st.selectbox("Семейный статус", list(personal_status_map.keys()))
        guarantors_rus = st.selectbox("Поручители/созаемщики", list(guarantors_map.keys()))
        residence_rus = st.selectbox("Срок проживания", list(residence_map.keys()))
        residence = residence_map[residence_rus]
        property_rus = st.selectbox("Тип имущества", list(property_map.keys()))
        other_installments_rus = st.selectbox("Другие рассрочки", list(other_installments_map.keys()))
        housing_rus = st.selectbox("Тип жилья", list(housing_map.keys()))
        existing_credits = st.slider("Действующие кредиты", 1, 4, 1)
        dependents_rus = st.selectbox("Количество иждивенцев", list(dependents_map.keys()))
        dependents = dependents_map[dependents_rus]
        phone_rus = st.selectbox("Подтвержденный телефон", list(phone_map.keys()))
        foreign_rus = st.selectbox("Иностранный заемщик", list(foreign_map.keys()))

    st.markdown("<br/>", unsafe_allow_html=True)
    run_button = st.button(" Узнать результат", type="primary", use_container_width=True)

if run_button:
    st.session_state.show_results = True

if st.session_state.show_results:
    row = build_input_df(
        age=age,
        amount=amount,
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
    color = "#17c964" if approved else "#f31260"

    st.markdown("<br/>", unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-title'>Наш вердикт</div>
                <div class='metric-value' style='color:{color};'>{decision}</div>
            </div>""", unsafe_allow_html=True
        )
    with col_b:
        prob_color = "#f31260" if pred >= threshold else "#17c964" if pred < threshold * 0.7 else "#f5a524"
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-title'>Оценка риска</div>
                <div class='metric-value' style='color:{prob_color};'>{pred*100:.1f}%</div>
            </div>""", unsafe_allow_html=True
        )
    with col_c:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-title'>Допустимый лимит</div>
                <div class='metric-value'>{threshold*100:.1f}%</div>
            </div>""", unsafe_allow_html=True
        )
    st.markdown("<br/>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([" Главный дашборд", " Инсайты и Факторы (SHAP)", " Моделирование (What-If)"])
    
    with tab1:

        explainer = CreditExplainer(trainer.model, X_train, feature_names)
        shap_info = explainer.explain(X_user)

        st.markdown("###  Объяснение инференса")
        st.write(generate_explanation(shap_info))

        brief = generate_human_brief(shap_info)
        st.markdown("###  Краткое резюме")
        st.markdown(
            f"**Уровень риска:** {brief['risk_level'].capitalize()}  "+
            f"| **Вероятность затруднений:** {brief['probability']*100:.1f}%"
        )
        st.markdown("**3 главные причины:**")
        for reason in brief["reasons"]:
            st.write(f"- {reason}")

        st.markdown("**Что можно сделать, чтобы снизить риск:**")
        for tip in brief["tips"]:
            st.write(f"- {tip}")

        st.markdown("###  Персональные рекомендации")
        auto_recs = generate_auto_recommendations(shap_info, amount, duration)
        for rec in auto_recs:
            st.markdown(f"<div class='info-panel'> {rec}</div>", unsafe_allow_html=True)

        st.markdown("###  Сегментный профиль")
        st.info("Анализ принадлежности к демографическим группам")
        seg_risk = generate_segment_risk(age, job_rus)
        st.markdown(f"**Группа:** {seg_risk['group']}")
        st.markdown(f"**Возрастной фактор:** {seg_risk['age_risk']}")
        st.markdown(f"**Профессиональный фактор:** {seg_risk['job_risk']}")

        with st.expander("❓ Как интерпретировать эти данные?", expanded=False):
            st.write(explain_waterfall_for_user())
            st.markdown(
                """
                Что важно смотреть в первую очередь:
                - Вероятность затруднений в процентах.
                - 2-3 фактора, которые сильнее всего повышают риск.
                - Можно ли снизить риск через what-if анализ (сумма и срок).
                """
            )

    with tab2:
        st.markdown("###  Влияние каждого фактора кредитного скоринга")
        friendly_feature_names = [humanize_feature_name(name) for name in shap_info["feature_names"]]
        top_idx = np.argsort(np.abs(shap_info["shap_values"]))[-8:][::-1]
        factors_df = pd.DataFrame(
            {
                "Фактор": [friendly_feature_names[i] for i in top_idx],
                "Влияние": ["повышает риск" if shap_info["shap_values"][i] > 0 else "снижает риск" for i in top_idx],
                "Сила влияния": [float(abs(shap_info["shap_values"][i])) for i in top_idx],
            }
        )
        st.dataframe(factors_df, width="stretch", hide_index=True)

        # SHAP waterfall intentionally left in codebase for technical analysis,
        # but hidden from the main UI to keep the product understandable for non-experts.
        if st.session_state.get("show_technical_shap_plot", False):
            fig = plt.figure(figsize=(9, 6))
            explainer.plot_waterfall(
                shap_values=shap_info["shap_values"],
                feature_names=friendly_feature_names,
                prediction=shap_info["prediction"],
                sample=shap_info["sample"],
            )
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        st.markdown("###  Регулировка параметров (What-if симуляция)")
        st.caption("Подвигайте ползунки, чтобы увидеть изменение вероятности дефолта.")
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
        st.write(
            f"Новая вероятность трудностей с выплатами: **{pred_whatif*100:.1f}%** "
            f"(изменение {delta*100:+.1f} п.п.)"
        )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "probability_default": round(pred, 6),
        "decision": decision,
        "threshold": round(threshold, 6),
        "what_if_probability_default": round(pred_whatif, 6),
        "what_if_delta_pp": round(delta * 100, 3),
        "client_input": row.to_dict(orient="records")[0],
    }
    st.download_button(
        "Скачать отчет по оценке (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="creditlens_report.json",
        mime="application/json",
        width="stretch",
    )

    st.subheader("Что мы планируем добавить в будущем")
    roadmap_df = pd.DataFrame(
        [
            {"Фича": "Загрузка анкеты из CRM", "Польза": "Исключает ручной ввод и ускоряет оценку", "Статус": "MVP-ready"},
            {"Фича": "Сегментный риск по профессиям/возрастам", "Польза": "Точнее политика одобрения", "Статус": "Реализовано"},
            {"Фича": "Авто-рекомендации клиенту", "Польза": "Показывает как повысить шанс одобрения", "Статус": "Реализовано"},
            {"Фича": "Мониторинг дрейфа качества", "Польза": "Контроль деградации модели в проде", "Статус": "Backlog"},
        ]
    )
    st.dataframe(roadmap_df, width="stretch", hide_index=True)
else:
    st.info("Пожалуйста, заполните данные в меню слева и нажмите «Узнать результат» — мы проанализируем ваш профиль.", icon=None)
