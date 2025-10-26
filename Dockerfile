FROM python:3.9-slim

WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание папок для данных и моделей
RUN mkdir -p data/raw data/processed models

# Экспоз порта для FastAPI
EXPOSE 8000

# Команда для запуска API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]