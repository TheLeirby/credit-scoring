from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List, Dict

app = FastAPI(title="Credit Scoring API", description="API для предсказания дефолта по кредиту")

# Загрузка модели при старте приложения
model = None

def load_model():
    """Загрузка обученной модели."""
    global model
    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

@app.on_event("startup")
async def startup_event():
    load_model()

# Pydantic модель для входных данных
class CreditApplication(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

class PredictionResult(BaseModel):
    prediction: int
    probability: float
    class_name: str

@app.post("/predict", response_model=PredictionResult)
async def predict(application: CreditApplication):
    """Предсказание дефолта для кредитной заявки."""
    try:
        # Преобразование входных данных в DataFrame
        input_data = pd.DataFrame([application.dict()])
        
        # Предсказание
        probability = model.predict_proba(input_data)[0, 1]
        prediction = int(probability > 0.5)  # Порог можно настроить
        
        class_name = "default" if prediction == 1 else "non-default"
        
        return PredictionResult(
            prediction=prediction,
            probability=probability,
            class_name=class_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Credit Scoring API работает!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}