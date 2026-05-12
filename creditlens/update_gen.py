# -*- coding: utf-8 -*-
import re

with open('creditlens/src/text_generator.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = """import random

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

    return random.choice(templates)"""

text = re.sub(r'def generate_explanation.*?return \([^)]+\)', replacement, text, flags=re.DOTALL)

if 'import random' not in text:
     text = text.replace('import numpy as np', 'import numpy as np\nimport random')

# Just making absolutely sure that `import random` doesn't get messed up if already present
text = text.replace('import random\n\nimport random', 'import random')

with open('creditlens/src/text_generator.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Generator updated")
