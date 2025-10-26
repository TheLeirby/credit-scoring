import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import pandas as pd
import numpy as np
import joblib
import os

from src.features.build_features import FeatureEngineer
from src.visualization.visualize import plot_roc_curve

def load_train_data():
    """Загрузка тренировочных данных."""
    return pd.read_csv("data/processed/train.csv")

def define_preprocessor():
    """Определение препроцессора для числовых и категориальных признаков."""
    numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 
                       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
                       'PAY_AMT5', 'PAY_AMT6']
    
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor

def train_model(experiment_name="Credit Scoring"):
    """Основная функция обучения модели с логированием в MLflow."""
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Загрузка данных
        train_df = load_train_data()
        X_train = train_df.drop("default", axis=1)
        y_train = train_df["default"]
        
        # Создание пайплайна
        preprocessor = define_preprocessor()
        
        pipeline = Pipeline(steps=[
            ('feature_engineer', FeatureEngineer()),
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Параметры для GridSearch
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        }
        
        # Поиск по сетке
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Логирование параметров и метрик
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Сохранение лучшей модели
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
        
        # Оценка на тестовых данных
        test_df = pd.read_csv("data/processed/test.csv")
        X_test = test_df.drop("default", axis=1)
        y_test = test_df["default"]
        
        y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
        y_pred = grid_search.best_estimator_.predict(X_test)
        
        # Вычисление метрик
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metrics({
            "test_roc_auc": roc_auc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        })
        
        # Логирование ROC-кривой
        roc_curve_plot = plot_roc_curve(y_test, y_pred_proba)
        mlflow.log_figure(roc_curve_plot, "roc_curve.png")
        
        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"ROC-AUC на тесте: {roc_auc:.4f}")
        
        return grid_search.best_estimator_

if __name__ == "__main__":
    train_model()