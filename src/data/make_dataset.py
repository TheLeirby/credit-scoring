import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path: str) -> pd.DataFrame:
    """Загрузка данных из CSV файла."""
    df = pd.read_csv(data_path)
    # У датасета есть небольшая особенность - первый столбец это ID, который стоит удалить
    df = df.drop("ID", axis=1)
    # Переименуем целевую переменную для удобства
    df = df.rename(columns={"default.payment.next.month": "default"})
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Базовая очистка данных."""
    # Проверим на явные дубликаты
    df = df.drop_duplicates()
    # В этом датасете пропусков нет, но на будущее:
    # df = df.dropna()
    return df

def save_data(df: pd.DataFrame, output_path: str):
    """Сохранение DataFrame в файл."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Загрузка и очистка
    df = load_data("data/raw/UCI_Credit_Card.csv")
    df = clean_data(df)
    
    # Разделение на train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["default"])
    
    # Сохранение
    save_data(train_df, "data/processed/train.csv")
    save_data(test_df, "data/processed/test.csv")
    print("Данные успешно подготовлены и разделены на train/test.")