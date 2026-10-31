
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.data_loader import load_data
from src.preprocess import build_preprocessor


def tune_model():
    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__class_weight': [None, 'balanced']
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print('Best parameters:', grid.best_params_)
    print('Best CV F1:', grid.best_score_)


if __name__ == '__main__':
    tune_model()
