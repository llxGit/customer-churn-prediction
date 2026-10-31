import joblib

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN,
    MODEL_PATH,
)

from src.data_loader import load_data
from src.preprocess import build_preprocessor

def train_model():
    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline = DummyClassifier(strategy='most_frequent')

    baseline.fit(X_train, y_train)

    baseline_pred = baseline.predict(X_test)
    
    model = Pipeline(
        steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(
            C=1,
            class_weight='balanced',
            max_iter=1000
            )),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
    
    print(f'acuracy score:{acc}\n')
    print(f'f1 score: {f1}\n')
    print(f'model path: {MODEL_PATH}\n')

    print(f'train rows: {len(X_train)}\n')
    print(f'test rows: {len(X_test)}\n')

    print("Baseline accuracy:", baseline_acc)
    print("Baseline F1:", baseline_f1)

    joblib.dump(model, MODEL_PATH)

if __name__ == '__main__':
    train_model()

   