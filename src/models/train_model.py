from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import yaml

def create_preprocessor(numeric_features, categorical_features):
    """Создание препроцессора для числовых и категориальных признаков."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def create_model_pipeline(model_type="logistic_regression", **hyperparams):
    """Создание полного пайплаина модели."""
    
    # Загрузка конфигурации
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    numeric_features = config['features']['numeric_features']
    categorical_features = config['features']['categorical_features']
    
    # Создание препроцессора
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Выбор классификатора
    if model_type == "logistic_regression":
        classifier = LogisticRegression(**hyperparams)
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(**hyperparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Полный пайплайн
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline