from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Класс для Feature Engineering."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # Пример feature engineering: создание агрегированного признака из истории платежей
        pay_columns = [f"PAY_{i}" for i in range(0, 7)]
        X['PAY_MEAN'] = X[pay_columns].mean(axis=1)
        X['PAY_STD'] = X[pay_columns].std(axis=1)
        
        # Биннинг возраста
        X['AGE_BINNED'] = pd.cut(X['AGE'], bins=[20, 30, 40, 50, 60, 100], labels=False)
        
        # Другие преобразования...
        # Логарифмирование баланса
        X['LIMIT_BAL_LOG'] = np.log1p(X['LIMIT_BAL'])
        
        return X

# Остальная предобработка (Imputer, Scaler, Encoder) будет в Sklearn Pipeline