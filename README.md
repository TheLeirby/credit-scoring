
# Credit Scoring Pipeline

Проект по автоматизации разработки и тестирования PD-модели для предсказания дефолта клиентов.

## Структура проекта


```
credit-scoring-model/
│
├── data/
│   ├── raw/                    # Исходные данные (версионируются DVC)
│   │   └── UCI_Credit_Card.csv
│   ├── processed/              # Подготовленные данные
│   │   ├── train.csv
│   │   └── test.csv
│   └── expectations/           # Наборы правил Great Expectations
│       ├── great_expectations.yml
│       └── credit_data_suite.json
│
├── models/                     # Сохраненные модели (версионируются DVC)
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── notebooks/                  # EDA и эксперименты
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/                   # Скрипты для обработки данных
│   │   ├── __init__.py
│   │   ├── make_dataset.py
│   │   └── validation.py       # Валидация с GE
│   ├── features/               # Feature Engineering
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/                 # Скрипты для обучения и предсказания
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── pipeline.py         # Создание sklearn pipeline
│   └── monitoring/
│       ├── __init__.py
│       └── drift_detection.py
│
├── api/                        # FastAPI приложение
│   ├── __init__.py
│   ├── app.py
│   └── schemas.py              # Pydantic схемы
│
├── tests/                      # Unit-тесты
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
│
├── .github/workflows/          # GitHub Actions
│   └── ci-cd.yml
│
├── scripts/
│   ├── train_pipeline.py
│   ├── run_monitoring.py
│   ├── build_docker.sh
│   └── run_docker.sh
│
├── config/                     # Конфигурационные файлы
│   ├── model_config.yaml
│   └── api_config.yaml
│
├── Dockerfile
├── requirements.txt
├── dvc.yaml                    # DVC pipeline
├── pyproject.toml              # Для конфигурации инструментов
├── params.yaml                 # Параметры для DVC pipeline
├── .env.example                # Пример переменных окружения
└── README.md
```




## Установка и запуск

1. Клонируйте репозиторий:


```bash
git clone https://github.com/leirby/credit-scoring-pipeline.git
cd credit-scoring-pipeline
```


2. Установите зависимости:

```bash
pip install -r requirements.txt
```
3. Запустите пайплайн подготовки данных и обучения:

```bash
dvc repro
```

4. Запустите API:

```bash
uvicorn api.app:app --reload
```

## Использование



### Обучение модели

```bash
python src/models/train_model.py
```

### Тестирование

```bash
pytest
```

### Запуск в Docker

```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
```

## Пример запроса к API

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, ...}'
```
