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
        "A12": "на счете от 0 до 10 000 ₽",
        "A13": "на счете более 10 000 ₽ или зарплатный счет более года",
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
        "A61": "сбережения меньше 5 000 ₽",
        "A62": "сбережения от 5 000 до 25 000 ₽",
        "A63": "сбережения от 25 000 до 50 000 ₽",
        "A64": "сбережения больше 50 000 ₽",
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


def generate_explanation(shap_dict: dict[str, Any]) -> str:
    """Генерирует человекочитаемое объяснение решения модели.
    
    NOTE: упрощенная версия, без random.choice из шаблонов — 
    делаем один последовательный текст, чтобы не звучало роботом.
    """
    shap_values = np.asarray(shap_dict["shap_values"])
    feature_names = list(shap_dict["feature_names"])
    prediction = float(shap_dict["prediction"])

    decision = "высокий риск (вероятен отказ)" if prediction >= 0.5 else "умеренный риск (вероятно одобрение)"

    pos_idx = np.argsort(shap_values)[-3:][::-1]
    neg_idx = np.argsort(shap_values)[:3]
    
    risk_factors: list[str] = []
    for i in pos_idx:
        if shap_values[i] > 0:
            risk_factors.append(_normalize_feature_name(feature_names[i]))

    protective_factors: list[str] = []
    for i in neg_idx:
        if shap_values[i] < 0:
            protective_factors.append(_normalize_feature_name(feature_names[i]))

    parts = [f"Итог оценки: {decision}. Вероятность проблем с выплатой — {prediction * 100:.1f}%."]
    
    if risk_factors:
        parts.append("Факторы, которые увеличили риск: " + ", ".join(risk_factors) + ".")
    if protective_factors:
        parts.append("Факторы, которые сработали в пользу клиента: " + ", ".join(protective_factors) + ".")

    return " ".join(parts)


def explain_waterfall_for_user() -> str:
    return (
        "График SHAP показывает, почему модель приняла именно такое решение. "
        "Каждая строка — отдельный фактор клиента. "
        "Красные полосы толкают оценку риска вверх, зеленые — вниз. "
        "Чем длиннее полоса, тем сильнее влияние. "
        "Самые важные причины решения — вверху графика."
    )


RECOMMENDATIONS = {
    "Amount": "Снизьте сумму кредита или увеличьте первоначальный взнос.",
    "Duration": "Подберите срок так, чтобы платеж не превышал 25% дохода.",
    "InstallmentRate": "Уменьшите долговую нагрузку: снизьте ежемесячный платеж или закройте часть обязательств.",
    "History": "Укрепите платежную дисциплину: избегайте просрочек в ближайшие 6–12 месяцев.",
    "Employment": "Подтвердите стабильную занятость справками и выписками.",
    "Status": "Предоставьте дополнительные финансовые подтверждения (выписки, поступления на счет).",
    "Savings": "Покажите накопления или финансовую подушку безопасности.",
    "ExistingCredits": "По возможности сократите число активных кредитов перед подачей заявки.",
    "Dependents": "Подтвердите дополнительные источники дохода семьи.",
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
        tips.append("Критичных факторов не выявлено. Поддерживайте текущую платежную дисциплину.")

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
        age_group = "Средний возраст (25–50 лет)"
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
        recommendations.append(f"Уменьшите запрашиваемую сумму примерно до {suggested_amount} ₽, чтобы снизить кредитную нагрузку.")
        
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
