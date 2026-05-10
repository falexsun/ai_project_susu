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
    "Savings": "Сбережения",
    "Guarantors": "Поручители",
    "Residence": "Срок проживания",
    "Property": "Имущество",
    "OtherInstallments": "Другие рассрочки",
    "ExistingCredits": "Количество действующих кредитов",
    "Dependents": "Количество иждивенцев",
    "Phone": "Наличие телефона",
    "Foreign": "Статус иностранного заемщика",
}


CODEBOOK = {
    "Status": {
        "A11": "на расчетном счете отрицательный остаток",
        "A12": "на счете от 0 до 200 DM",
        "A13": "на счете более 200 DM или зарплатный счет более года",
        "A14": "расчетного счета нет",
    },
    "History": {
        "A30": "ранее кредиты брались и погашались вовремя",
        "A31": "все кредиты в этом банке закрыты вовремя",
        "A32": "текущие кредиты пока погашаются без просрочек",
        "A33": "были задержки платежей в прошлом",
        "A34": "критичная кредитная история или есть другие проблемные кредиты",
    },
    "Purpose": {
        "A40": "автомобиль (новый)",
        "A41": "автомобиль (подержанный)",
        "A42": "мебель или оборудование",
        "A43": "бытовая техника (радио/ТВ)",
        "A44": "домашние приборы",
        "A45": "ремонт",
        "A46": "образование",
        "A47": "отпуск",
        "A48": "переобучение",
        "A49": "бизнес",
        "A410": "прочие цели",
    },
    "Savings": {
        "A61": "сбережения меньше 100 DM",
        "A62": "сбережения от 100 до 500 DM",
        "A63": "сбережения от 500 до 1000 DM",
        "A64": "сбережения больше 1000 DM",
        "A65": "сбережения неизвестны или отсутствуют",
    },
    "Employment": {
        "A71": "безработный",
        "A72": "стаж меньше 1 года",
        "A73": "стаж от 1 до 4 лет",
        "A74": "стаж от 4 до 7 лет",
        "A75": "стаж более 7 лет",
    },
    "PersonalStatus": {
        "A91": "мужчина, разведен или живет отдельно",
        "A92": "женщина, разведена или замужем",
        "A93": "мужчина, не женат",
        "A94": "мужчина, женат или вдовец",
        "A95": "женщина, не замужем",
    },
    "Guarantors": {
        "A101": "поручителей нет",
        "A102": "есть созаемщик",
        "A103": "есть поручитель",
    },
    "Property": {
        "A121": "есть недвижимость",
        "A122": "есть накопления или страхование жизни",
        "A123": "есть автомобиль или другое имущество",
        "A124": "имущество не подтверждено или отсутствует",
    },
    "OtherInstallments": {
        "A141": "есть рассрочки/обязательства в банке",
        "A142": "есть рассрочки в магазинах",
        "A143": "других рассрочек нет",
    },
    "Housing": {
        "A151": "жилье арендуется",
        "A152": "собственное жилье",
        "A153": "проживание бесплатно",
    },
    "Job": {
        "A171": "безработный или неквалифицированный нерезидент",
        "A172": "неквалифицированный работник",
        "A173": "квалифицированный сотрудник",
        "A174": "руководитель, предприниматель или высококвалифицированный сотрудник",
    },
    "Phone": {
        "A191": "личный телефон не указан",
        "A192": "личный телефон подтвержден",
    },
    "Foreign": {
        "A201": "иностранный заемщик",
        "A202": "не иностранный заемщик",
    },
}


def _decode_code(field_name: str, raw_value: str) -> str:
    decoded = CODEBOOK.get(field_name, {}).get(raw_value)
    if decoded is None:
        return raw_value
    return decoded


def humanize_feature_name(feature_name: str) -> str:
    raw = feature_name.split("__")[-1]

    if "_A" in raw:
        field, value = raw.split("_", 1)
        field_label = FEATURE_TEMPLATES.get(field, field)
        value_label = _decode_code(field, value)
        return f"{field_label}: {value_label}"

    field = raw.split("_")[0]
    return FEATURE_TEMPLATES.get(field, raw)


def _normalize_feature_name(feature_name: str) -> str:
    return humanize_feature_name(feature_name)


import random

def generate_explanation(shap_dict: dict[str, Any]) -> str:
    shap_values = np.asarray(shap_dict["shap_values"])
    feature_names = list(shap_dict["feature_names"])
    prediction = float(shap_dict["prediction"])

    decision = "высокий риск (вероятен отказ)" if prediction >= 0.5 else "умеренный риск (вероятно одобрение)"

    pos_idx = np.argsort(shap_values)[-4:][::-1]
    neg_idx = np.argsort(shap_values)[:4]
    
    risk_factors: list[str] = []
    for i in pos_idx:
        if shap_values[i] > 0:
            name = _normalize_feature_name(feature_names[i])
            risk_factors.append(f"{name} (повышает риск)")

    protective_factors: list[str] = []
    for i in neg_idx:
        if shap_values[i] < 0:
            name = _normalize_feature_name(feature_names[i])
            protective_factors.append(f"{name} (снижает риск)")

    risk_text = "; ".join(risk_factors) if risk_factors else "существенных риск-факторов не найдено"
    prot_text = "; ".join(protective_factors) if protective_factors else "положительных факторов мало"

    confidence = "высокая" if prediction >= 0.7 or prediction <= 0.3 else "средняя"
    
    templates = [
        (
            f"Итог оценки: {decision}. "
            f"Риск затруднений при выплате мы оцениваем в {prediction * 100:.1f}%. "
            f"Уверенность нашей системы: {confidence}. "
            f"Что увеличило риск: {risk_text}. "
            f"Что помогло снизить риск: {prot_text}."
        ),
        (
            f"На основе ваших данных мы прогнозируем {decision}. "
            f"Модель оценивает вероятность проблем с кредитом в {prediction * 100:.1f}% (уверенность: {confidence}). "
            f"Основные факторы риска: {risk_text}. "
            f"Позитивные факторы: {prot_text}."
        ),
        (
            f"Система проанализировала анкету и выявила {decision} с вероятностью {prediction * 100:.1f}%. "
            f"Степень уверенности алгоритма: {confidence}. "
            f"Обратите внимание на эти моменты: {risk_text}. "
            f"Смягчающие обстоятельства: {prot_text}."
        ),
        (
             f"Вердикт алгоритма: {decision}. "
             f"Шанс возникновения трудностей с платежами около {prediction * 100:.1f}% (с уверенностью уровня '{confidence}'). "
             f"Главные барьеры для одобрения: {risk_text}. "
             f"Ваши сильные стороны: {prot_text}."
        )
    ]

    return random.choice(templates)


def explain_waterfall_for_user() -> str:
    return (
        "Как читать график SHAP простыми словами: "
        "график показывает, почему модель выдала именно такой риск. "
        "Каждая строка - это один фактор клиента. "
        "Красные факторы двигают риск возникновения трудностей с выплатой вверх (хуже), "
        "зеленые - вниз (лучше). "
        "Чем длиннее полоска, тем сильнее влияние этого фактора. "
        "Верхние строки на графике - самые важные причины решения."
    )


RECOMMENDATIONS = {
    "Amount": "Снизить сумму кредита или увеличить первоначальный взнос.",
    "Duration": "Подобрать более комфортный срок кредита и платеж под доход.",
    "InstallmentRate": "Уменьшить долговую нагрузку: снизить ежемесячный платеж или закрыть часть обязательств.",
    "History": "Укрепить платежную дисциплину: избегать просрочек в ближайшие месяцы.",
    "Employment": "Подтвердить стабильную занятость и стаж документами.",
    "Status": "Предоставить дополнительные финансовые подтверждения (выписки, регулярные поступления).",
    "Savings": "Показать накопления или финансовую подушку.",
    "ExistingCredits": "По возможности сократить число активных кредитов перед новой заявкой.",
    "Dependents": "Подтвердить дополнительные источники дохода семьи.",
}


def _base_feature_key(feature_name: str) -> str:
    raw = feature_name.split("__")[-1]
    return raw.split("_")[0]


def generate_human_brief(shap_dict: dict[str, Any]) -> dict[str, Any]:
    shap_values = np.asarray(shap_dict["shap_values"])
    feature_names = list(shap_dict["feature_names"])
    prediction = float(shap_dict["prediction"])

    ordered = np.argsort(np.abs(shap_values))[::-1]
    top_idx = ordered[:3]

    reasons: list[str] = []
    tips: list[str] = []
    for i in top_idx:
        feature = feature_names[i]
        direction = "повышает" if shap_values[i] > 0 else "снижает"
        label = humanize_feature_name(feature)
        reasons.append(f"{label} {direction} риск")

        key = _base_feature_key(feature)
        rec = RECOMMENDATIONS.get(key)
        if rec and shap_values[i] > 0:
            tips.append(rec)

    if not tips:
        tips.append("Критичных факторов не видно. Поддерживайте текущую платежную дисциплину.")

    risk_level = "высокий" if prediction >= 0.65 else "средний" if prediction >= 0.35 else "низкий"

    return {
        "risk_level": risk_level,
        "probability": prediction,
        "reasons": reasons,
        "tips": tips[:3],
    }

def generate_segment_risk(age: int, job_type: str) -> dict[str, str]:
    if age < 25:
        age_group = "Молодежь (до 25 лет)"
        age_risk = "Повышенный риск из-за потенциальной нестабильности доходов."
    elif 25 <= age <= 50:
        age_group = "Средний возраст (25-50 лет)"
        age_risk = "Минимальный риск, как правило, наиболее стабильная платежеспособность."
    else:
        age_group = "Старший возраст (старше 50 лет)"
        age_risk = "Умеренный риск, зависит от пенсионного статуса и накоплений."

    if "высококвалифицированный" in job_type or "руководитель" in job_type:
        job_risk_desc = "Низкий риск: высокая стабильность и уровень дохода."
    elif "квалифицированный" in job_type:
        job_risk_desc = "Обычный риск: стандартная занятость."
    elif "безработный" in job_type:
        job_risk_desc = "Высокий риск: отсутствие регулярного трудового дохода."
    else:
        job_risk_desc = "Повышенный риск: возможна нестабильность занятости."

    return {
        "group": f"{age_group} | Профессия: {job_type}",
        "age_risk": age_risk,
        "job_risk": job_risk_desc
    }

def generate_auto_recommendations(shap_dict: dict[str, Any], amount: int, duration: int) -> list[str]:
    shap_values = np.asarray(shap_dict["shap_values"])
    feature_names = list(shap_dict["feature_names"])
    
    recommendations = []
    
    # Ищем, влияет ли сумма и срок негативно
    amount_idx = next((i for i, name in enumerate(feature_names) if "Amount" in name), -1)
    duration_idx = next((i for i, name in enumerate(feature_names) if "Duration" in name), -1)
    
    if amount_idx != -1 and shap_values[amount_idx] > 0:
        suggested_amount = int(amount * 0.8)
        recommendations.append(f"Уменьшите запрашиваемую сумму примерно до {suggested_amount} DM, чтобы снизить кредитную нагрузку.")
        
    if duration_idx != -1 and shap_values[duration_idx] > 0:
        suggested_duration = min(72, int(duration * 1.2) if duration < 48 else max(6, int(duration * 0.8)))
        recommendations.append(f"Рассмотрите изменение срока кредита на {suggested_duration} мес., чтобы сбалансировать платежи.")
        
    # Добавляем общие советы по топ-рискам
    ordered = np.argsort(np.abs(shap_values))[::-1]
    for i in ordered:
        if shap_values[i] > 0 and len(recommendations) < 4:
            feat_key = _base_feature_key(feature_names[i])
            rec = RECOMMENDATIONS.get(feat_key)
            if rec and rec not in recommendations and "сумму" not in rec.lower() and "срок" not in rec.lower():
                recommendations.append(rec)
                
    if not recommendations:
        recommendations.append("Особых рекомендаций нет, текущие параметры заявки выглядят оптимально.")
        
    return recommendations
