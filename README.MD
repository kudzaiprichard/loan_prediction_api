# Loan Prediction API Documentation

This project is an API for predicting loan approval using various machine learning models. The API is built using FastAPI and allows users to train models on a provided dataset and make predictions using the trained models.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Endpoints](#endpoints)
  - [Train Models Endpoint](#1-train-models-endpoint)
  - [Predict Endpoint](#2-predict-endpoint)
- [Conclusion](#conclusion)

## Project Structure

``` markdown
loan_prediction_api/
│
├── app/
│   ├── data/
│   │   └── clean_data.csv
│   │   
│   ├── models/
│   │   ├── dataset/
│   │   │   └── data_science_competition_2024.csv
│   │   └── probability_of_default.ipynb
│   │   
│   ├── routes/
│   │   ├── predict.py
│   │   ├── train.py
│   │   └── location_handler.py
│   │   
│   └── main.py
│
├── requirements.txt
└── README.md

```

- **app/**: Contains the main application files.
  - **main.py**: The entry point of the application.
  - **data/**: Directory for storing clean dataset to train models via api.
      - **clean_data.csv**: The cleaned dataset exported from notebook.
  - **models/**: Directory containing notebook and for storing trained model files.
      - **dataset/**: Directory containing original uncleaned dataset.
          - **data_science_competition_2024.csv**: The uncleaned original dataset.
      - **probability_of_default.ipynb**: Jupyter notebook for exploratory data analysis (EDA), data cleaning and training model.
  - **routes/**: Directory containing all the restfull endpoints for the system.
      - **train.py**: Python file to handle application train requests and responses.
      - **predict.py**: Python file to handle application predicts requests and responses.
      - **location_handler.py**: Python file managing location data and requests.
- **requirements.txt**: Lists the project dependencies.
- **README.md**: Project documentation


## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository**:

```bash
git clone https://github.com/kudzaiprichard/loan_prediction_api.git
cd loan_prediction_api
 ```
   
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the API**:

```bash
uvicorn app.main:app --reload
```

The base URL for all endpoints is `http://localhost:8000`

## Endpoints

### 1. Train Models Endpoint

- **URL**: `/train`
- **Method**: POST
- **Description**: Trains multiple machine learning models for loan prediction using the provided dataset.
- **Response**: Returns JSON with training results for each model.

#### Example Request

```bash
curl -X 'POST' \
  'http://localhost:8000/train' \
  -H 'Content-Type: application/json' 
```

#### Example Response

```json
{
	"accuracy": "90.32 %",
	"precision": "90.32 %",
	"recall": "90.32 %",
	"f1_score": "90.32 %"
}
```

### 2. Predict Endpoint

- **URL**: `/predict`
- **Method**: POST
- **Description**: Predicts loan approval using the random forest model.
- **Request Body**: JSON object containing loan application data.
- **Response**: Returns JSON with the prediction result or an error if there is any validation problems.

#### Example Requests

**Successful Prediction**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
        "salary": 50000.0,
        "location": "Harare",
        "remaining_term": 10.0,
        "age": 35,
        "interest_rate": 5.0,
        "marital_status": 1,
        "is_employed": 1
      }'
```

**Invalid Age**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict?model_name=Invalid%20Model' \
  -H 'Content-Type: application/json' \
  -d '{
        "salary": 50000.0,
        "location": "Harare",
        "remaining_term": 10.0,
        "age": 35,
        "interest_rate": 5.0,
        "marital_status": 1,
        "is_employed": 1
      }'
```

#### Example Responses

**Successful Prediction Response**

```json
{
  "prediction": "Loan will not default"
}
```

**Invalid Age Response**

```json
{
	"detail": [
		{
			"type": "value_error",
			"loc": [
				"body",
				"age"
			],
			"msg": "Value error, Age must be between 18 and 100.",
			"input": 200.0,
			"ctx": {
				"error": {}
			}
		}
	]
}
```

## Conclusion

This documentation provides an overview of the project structure, how to get started, and detailed explanations of the API endpoints.

