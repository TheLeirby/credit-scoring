import pandas as pd
import pytest
from src.data.make_dataset import load_data, clean_data

def test_load_data():
    """Тест загрузки данных."""
    # Создаем временный CSV для теста
    test_data = pd.DataFrame({
        "ID": [1, 2],
        "LIMIT_BAL": [10000, 20000],
        "default.payment.next.month": [0, 1]
    })
    test_data.to_csv("test_data.csv", index=False)
    
    df = load_data("test_data.csv")
    assert "ID" not in df.columns
    assert "default" in df.columns
    
    # Удаляем временный файл
    import os
    os.remove("test_data.csv")

def test_clean_data():
    """Тест очистки данных."""
    df = pd.DataFrame({
        "LIMIT_BAL": [10000, 10000, 20000],
        "default": [0, 0, 1]
    })
    cleaned_df = clean_data(df)
    assert len(cleaned_df) == 2  # Дубликаты удалены