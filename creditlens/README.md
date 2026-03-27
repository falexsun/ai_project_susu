# CreditLens — Интерпретируемый кредитный скоринг

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

## Описание проекта
CreditLens — это практическая система кредитного скоринга, которая прогнозирует риск дефолта клиента по данным анкеты и кредитной истории. Проект ориентирован не только на качество прогноза, но и на прозрачность решений для аналитиков, риск-менеджеров и бизнеса.

Ключевая идея — объединить нейронную сеть для классификации с SHAP-объяснениями, чтобы каждое решение можно было интерпретировать на уровне конкретных факторов. Это помогает обосновывать отказ или одобрение заявки в понятной форме.

В рамках проекта реализован полный ML-цикл: загрузка данных, предобработка, сравнение baseline-моделей, обучение MLP, визуализация SHAP и интерактивное Streamlit-демо с what-if анализом.

## Архитектура решения
Поток данных:

`Данные (UCI/Kaggle) → Предобработка (импутация/кодирование/масштабирование) → MLP (CreditNet) → SHAP-анализ → Текстовое объяснение`

Полная модель для прод-скоринга:

`LogReg + GradientBoosting + MLP (CreditNet) → Stacking meta-model (LogReg) → p(default)`

## Результаты моделей
Итоговая таблица метрик на тестовой выборке:

| Модель          | ROC-AUC | PR-AUC | F1  |
|-----------------|---------|--------|-----|
| LogReg          | 0.819   | 0.688  | 0.658 |
| XGBoost/GB      | 0.777   | 0.627  | 0.592 |
| MLP (CreditNet) | 0.816   | 0.659  | 0.654 |

## Датасеты
- `german` (UCI Statlog German Credit): 1000 записей, 20 признаков
- `uci_credit_card` (UCI Default of Credit Card Clients): 30000 записей, 23 признака
- `give_me_some_credit` (Kaggle): загрузка из локального CSV после скачивания
- `home_credit` (Kaggle Home Credit): загрузка из локального CSV после скачивания

По умолчанию приложение Streamlit использует артефакты датасета `german`.

## Установка и запуск
```bash
pip install -r requirements.txt

# German (по умолчанию)
python src/download_data.py --dataset german
python src/preprocess.py --dataset german
python src/model.py --dataset german
python src/ensemble.py --dataset german

# UCI Credit Card
python src/download_data.py --dataset uci_credit_card
python src/preprocess.py --dataset uci_credit_card
python src/model.py --dataset uci_credit_card --epochs 40
python src/ensemble.py --dataset uci_credit_card

# Kaggle наборы (после ручного скачивания CSV в data/raw/kaggle/...)
python src/download_data.py --dataset give_me_some_credit
python src/preprocess.py --dataset give_me_some_credit
python src/model.py --dataset give_me_some_credit
python src/ensemble.py --dataset give_me_some_credit

python src/download_data.py --dataset home_credit
python src/preprocess.py --dataset home_credit
python src/model.py --dataset home_credit
python src/ensemble.py --dataset home_credit

streamlit run app.py

# API сервис
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## API: что отправляет клиент
Перед расчетом получите обязательные поля для датасета:

`GET /schema/{dataset}`

Пример запроса клиента:

```json
{
  "dataset": "german",
  "client_data": {
    "Status": "A14",
    "Duration": 24,
    "History": "A32",
    "Purpose": "A43",
    "Amount": 5000,
    "Savings": "A61",
    "Employment": "A173",
    "InstallmentRate": 2,
    "PersonalStatus": "A93",
    "Guarantors": "A101",
    "Residence": 2,
    "Property": "A123",
    "Age": 35,
    "OtherInstallments": "A143",
    "Housing": "A152",
    "ExistingCredits": 1,
    "Job": "A173",
    "Dependents": 1,
    "Phone": "A192",
    "Foreign": "A201"
  }
}
```

Ответ API:
- `probability_default`: вероятность дефолта
- `threshold`: порог решения
- `decision`: `APPROVE` или `REJECT`

## Структура проекта
```text
creditlens/
  data/
    raw/
    processed/
  notebooks/
    01_eda.ipynb
    02_baseline.ipynb
    03_neural_net.ipynb
    04_shap.ipynb
  src/
    download_data.py
    preprocess.py
    model.py
    explainer.py
    text_generator.py
  app.py
  requirements.txt
  README.md
```

## Технологии
- PyTorch
- SHAP
- scikit-learn
- Streamlit
- pandas
