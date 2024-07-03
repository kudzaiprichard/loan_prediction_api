# app/routes/predict.py

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
import joblib
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

router = APIRouter()


class LoanApplication(BaseModel):
    sex: int
    age: int
    job: int
    location: int
    marital_status: int
    is_employed: int
    salary: float
    loan_amount: float
    outstanding_balance: float
    remaining_term: float
    number_of_defaults: int
    loan_to_income_ratio: float
    is_delinquent: int
    year: int
    month: int
    day: int
    quarter: int
    day_of_week: int
    interest_rate: float


@router.post("/predict")
def predict_default(application: LoanApplication,
                    model_name: str = Query(..., description="Name of the model to use for prediction")):
    model_path = f"models/{model_name.replace(' ', '_').lower()}.pkl"
    print(model_path)
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Please train the model first.")

    input_data = pd.DataFrame([application.dict().values()], columns=application.dict().keys())

    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        return {"prediction": "Loan will not default"}
    else:
        return {"prediction": "Loan will default"}
