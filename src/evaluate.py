import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import (
    MODEL_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    METRICS_PATH,
)

from src.data_loader import load_data

def evaluate_model():
    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recal = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    report = classification_report(y_test, y_pred, zero_division=0)
    matrix =confusion_matrix(y_test, y_pred)

    output = (
        f'acc: {acc}\n'
        f'precision: {precision}\n'
        f'recal: {recal}\n'
        f'f1: {f1}\n'
        f'report: {report}\n'
        f'matrix: {matrix}\n'
    )
    print(output)


    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.write_text(
        output,
        encoding='utf-8',
    )

    print(f'Metrics saved: {METRICS_PATH}')


if __name__ == '__main__':
    evaluate_model()    