
# Credit Scoring Pipeline

Проект по автоматизации разработки и тестирования PD-модели для предсказания дефолта клиентов.

## Структура проекта






credit-scoring-pipeline/
│
├── data/
│   ├── raw/                    # Исходные данные (управляются DVC)
│   ├── processed/              # Обработанные данные
│   └── great_expectations/     # Настройки и сьюты Great Expectations
│
├── models/                     # Сохраненные модели (управляются DVC)
│
├── notebooks/                  # Jupyter notebooks для исследования
│   └── 01_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py    # Скрипт загрузки и очистки
│   │   └── validation.py      # Great Expectations валидация
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py  # Feature Engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── predict_model.py   # Функции для предсказания
│   │   └── train_model.py     # Обучение и логирование в MLflow
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py       # Построение графиков (ROC-кривая)
│
├── api/
│   ├── __init__.py
│   └── app.py                 # FastAPI приложение
│
├── tests/                     # Unit-тесты
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── .github/workflows/         # GitHub Actions CI
│   └── ci-cd.yml
│
├── scripts/                   # Вспомогательные скрипты
│   ├── train_pipeline.py      # Скрипт для запуска всего пайплайна
│   ├── monitor_drift.py       # Скрипт для мониторинга дрифта (опционально)
│   ├── build_docker.sh
│   └── run_docker.sh
│
├── Dockerfile
├── requirements.txt
├── dvc.yaml                   # DVC пайплайн
├── params.yaml                # Параметры для обучения (гиперпараметры)
└── README.md






## Установка и запуск

1. Клонируйте репозиторий:



git clone https://github.com/leirby/credit-scoring-pipeline.git
cd credit-scoring-pipeline



2. Установите зависимости:


pip install -r requirements.txt

3. Запустите пайплайн подготовки данных и обучения:

dvc repro


4. Запустите API:

uvicorn api.app:app --reload


## Использование



### Обучение модели

python src/models/train_model.py


### Тестирование

pytest

### Запуск в Docker


docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api


## Пример запроса к API

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, ...}'
