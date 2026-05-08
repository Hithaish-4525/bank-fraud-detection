from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

# Get the correct path to the models
# This goes UP one level and then into models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'fraud_model.pkl')

# Load the model
model = joblib.load(model_path)

class Transaction(BaseModel):
    # The Kaggle model needs 30 features (V1-V28 + Time + Amount)
    features: List[float]

@app.get("/")
def home():
    return {"message": "Fraud Detection API is Online"}

@app.post("/predict")
def predict(data: Transaction):
    # Prepare data for model
    # The model expects a DataFrame with specific column names
    column_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    df = pd.DataFrame([data.features], columns=column_names)
    
    # Get probability
    prob = model.predict_proba(df)[0][1]
    
    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": bool(prob > 0.5),
        "status": "High Risk" if prob > 0.7 else "Safe"
    }

