import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from .pipeline import create_model_pipeline
from src.features.build_features import FeatureEngineer

def load_config():
    """Загрузка конфигурации."""
    with open('config/model_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_model():
    """Основная функция обучения модели."""
    config = load_config()
    
    # Настройка MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Загрузка данных
    train_df = pd.read_csv(f"{config['data']['processed_dir']}/train.csv")
    X_train = train_df.drop(config['features']['target'], axis=1)
    y_train = train_df[config['features']['target']]
    
    with mlflow.start_run():
        # Feature Engineering
        feature_engineer = FeatureEngineer()
        X_train_processed = feature_engineer.fit_transform(X_train)
        
        # Создание пайплайна
        pipeline = create_model_pipeline(
            model_type=config['model']['classifier']
        )
        
        # Подбор гиперпараметров
        grid_search = GridSearchCV(
            pipeline,
            config['model']['hyperparameters'][config['model']['classifier']],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_processed, y_train)
        
        # Логирование в MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Оценка на тестовых данных
        test_df = pd.read_csv(f"{config['data']['processed_dir']}/test.csv")
        X_test = test_df.drop(config['features']['target'], axis=1)
        y_test = test_df[config['features']['target']]
        
        X_test_processed = feature_engineer.transform(X_test)
        y_pred_proba = grid_search.best_estimator_.predict_proba(X_test_processed)[:, 1]
        y_pred = grid_search.best_estimator_.predict(X_test_processed)
        
        # Вычисление метрик
        metrics = {
            "test_roc_auc": roc_auc_score(y_test, y_pred_proba),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred)
        }
        
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
        
        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"ROC-AUC на тесте: {metrics['test_roc_auc']:.4f}")
        
        return grid_search.best_estimator_

if __name__ == "__main__":
    train_model()