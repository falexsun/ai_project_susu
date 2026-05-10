# CreditLens — Makefile
# Курсовой проект по дисциплине "Нейронные сети и искусственный интеллект"

PYTHON := python3
PIP := pip3
DATASET ?= german

.PHONY: help install data preprocess train ensemble benchmark test api app clean

help:
	@echo "CreditLens — доступные команды:"
	@echo "  make install      — установка зависимостей"
	@echo "  make data         — загрузка датасета (DATASET=german|uci_credit_card|...)"
	@echo "  make preprocess   — предобработка данных"
	@echo "  make train        — обучение нейронной сети"
	@echo "  make ensemble     — обучение ансамбля"
	@echo "  make benchmark    — сравнение всех моделей"
	@echo "  make test         — запуск unit-тестов"
	@echo "  make api          — запуск FastAPI сервиса"
	@echo "  make app          — запуск Streamlit приложения"
	@echo "  make all          — полный пайплайн: data -> preprocess -> train -> ensemble -> benchmark"
	@echo "  make clean        — очистка артефактов"

install:
	$(PIP) install -r requirements.txt
	@echo "[make] Зависимости установлены."

data:
	$(PYTHON) src/download_data.py --dataset $(DATASET)

preprocess:
	$(PYTHON) src/preprocess.py --dataset $(DATASET)

train:
	$(PYTHON) src/model.py --dataset $(DATASET)

ensemble:
	$(PYTHON) src/ensemble.py --dataset $(DATASET)

benchmark:
	$(PYTHON) src/benchmark.py --dataset $(DATASET)

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

api:
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app.py

all: data preprocess train ensemble benchmark
	@echo "[make] Полный пайплайн завершен для датасета '$(DATASET)'."

clean:
	find data/processed -name "*.npy" -delete
	find data/processed -name "*.pkl" -delete
	find data/processed -name "*.pt" -delete
	find data/processed -name "*.json" -delete
	find data/plots -name "*.png" -delete
	@echo "[make] Артефакты очищены."
