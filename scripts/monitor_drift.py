import pandas as pd
import numpy as np
import requests
import json
from scipy.stats import ks_2samp

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI)."""
    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Избегаем деления на 0
    expected_percents = np.where(expected_percents == 0, 0.001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.001, actual_percents)
    
    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi

def monitor_drift():
    """Мониторинг дрифта данных и предсказаний."""
    # Загрузка тренировочных данных (эталон)
    train_df = pd.read_csv("data/processed/train.csv")
    
    # Имитация новых данных (берем часть тестовой выборки)
    test_df = pd.read_csv("data/processed/test.csv")
    new_data = test_df.sample(1000, random_state=42)
    
    # Получение предсказаний от API
    predictions = []
    for _, row in new_data.iterrows():
        # Отправка запроса к API
        # (в реальности лучше батч-запросы или асинхронные вызовы)
        data = row.drop("default").to_dict()
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            predictions.append(result["probability"])
    
    # Расчет PSI для распределения вероятностей
    train_probas = [...]  # Здесь должны быть вероятности на тренировочных данных
    psi = calculate_psi(train_probas, predictions)
    
    print(f"PSI: {psi:.4f}")
    
    if psi > 0.25:
        print("ВНИМАНИЕ: Значительный дрифт обнаружен!")
    elif psi > 0.1:
        print("Предупреждение: умеренный дрифт обнаружен.")
    else:
        print("Дрифт в пределах нормы.")

if __name__ == "__main__":
    monitor_drift()