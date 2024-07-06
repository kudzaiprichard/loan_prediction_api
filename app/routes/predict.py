# app/routes/predict.py

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
import joblib
from pydantic import BaseModel, validator, Field
from sklearn.preprocessing import StandardScaler
from .location_handler import get_location_id  # Import the location handling function

router = APIRouter()


class LoanApplication(BaseModel):
    salary: float = Field(..., example=5000.0, description="Salary of the loan applicant")
    location: str = Field(..., example="Harare", description="Location of the loan applicant")
    remaining_term: float = Field(..., example=10.0, description="Remaining loan term in years")
    age: int = Field(..., example=30, description="Age of the loan applicant")
    interest_rate: float = Field(..., example=5.0, description="Interest rate on the loan")
    marital_status: int = Field(..., example=1, description="Marital status (0 - not married, 1 - married)")
    is_employed: int = Field(..., example=1, description="Employment status (0 - unemployed, 1 - employed)")

    @validator('salary', 'remaining_term', 'interest_rate')
    def check_positive(cls, value):
        if value <= 0:
            raise ValueError('Salary, remaining term, and interest rate must be positive.')
        return value

    @validator('age')
    def check_age(cls, value):
        if not 18 <= value <= 100:
            raise ValueError('Age must be between 18 and 100.')
        return value

    @validator('marital_status', 'is_employed')
    def check_binary_value(cls, value):
        if value not in [0, 1]:
            raise ValueError('Marital status and employment status must be 0 or 1.')
        return value

    @validator('location')
    def check_location(cls, value):
        location_id = get_location_id(value)
        if location_id == 25:  # Assuming 25 is the default location ID
            raise ValueError('Invalid location provided.')
        return value


# Load the trained Random Forest model
model = joblib.load("models/random_forest_model.pkl")


@router.post("/predict")
def predict_default(application: LoanApplication):
    # Validate LoanApplication fields
    try:
        application = LoanApplication(**application.dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Get location ID using location_handler module
    location_id = get_location_id(application.location)

    # Ensure only allowed parameters are used
    allowed_parameters = ['salary', 'location', 'remaining_term', 'age', 'interest_rate', 'marital_status',
                          'is_employed']
    application_dict = application.dict()
    application_dict['location'] = location_id  # Update location to integer ID

    filtered_data = {key: application_dict[key] for key in allowed_parameters}

    input_data = pd.DataFrame([filtered_data])

    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        return {"prediction": "Loan will not default"}
    else:
        return {"prediction": "Loan will default"}
