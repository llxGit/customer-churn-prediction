import pandas as pd

from src.config import RAW_DATA_PATH

def load_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f'file not found ;[     : {RAW_DATA_PATH}'
        )

    df = pd.read_csv(RAW_DATA_PATH)

    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    df = df.dropna(subset=['TotalCharges'])
    df = df.drop(columns=['customerID'])

    df['Churn'] = df['Churn'].map({
        'No': 0,
        'Yes': 1
    })

    return df