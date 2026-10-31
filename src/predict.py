import joblib
import pandas as pd

from src.config import MODEL_PATH


def predict_customer(customer: dict) -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'fail was not found ;( {MODEL_PATH}'
        )
    

    model = joblib.load(MODEL_PATH)

    customer_df = pd.DataFrame([customer])

    y_pred = model.predict(customer_df)

    return int(y_pred[0])

if __name__ == '__main__':
    new_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.0,
    "TotalCharges": 180.0,
} 

    prediction_new = predict_customer(new_customer)
    print(prediction_new)