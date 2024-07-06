# app/routes/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
from fastapi import APIRouter, HTTPException

router = APIRouter()


def evaluate_preds(y_true, y_preds):
    accuracy = round(accuracy_score(y_true, y_preds) * 100, 2)
    precision = round(precision_score(y_true, y_preds, average="micro") * 100, 2)
    recall = round(recall_score(y_true, y_preds, average="micro") * 100, 2)
    f1 = round(f1_score(y_true, y_preds, average="micro") * 100, 2)
    return accuracy, precision, recall, f1


@router.post("/train")
def train_random_forest():
    data = pd.read_csv("data/training_data.csv")
    X = data.drop(columns=['loan_status'])
    y = data['loan_status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate predictions
    y_preds = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate_preds(y_test, y_preds)

    # Save the model
    joblib.dump(model, "models/random_forest_model.pkl")

    results = {
        "accuracy": f"{accuracy} %",
        "precision": f"{precision} %",
        "recall": f"{recall} %",
        "f1_score": f"{f1} %"
    }

    return results
