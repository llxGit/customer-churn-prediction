
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from src.data_loader import load_data
from src.preprocess import build_preprocessor


def compare_models():

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

    models = {
        'Baseline': DummyClassifier(strategy='most_frequent'),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=RANDOM_STATE
        ),


        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = {}

    for name, classifier in models.items():
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', classifier)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )

        results[name] = scores.mean()

        print(f'{name}: {scores.mean():.4f}')

    sorted_results = sorted(
        results.items(),
        key=lambda item: item[1],
        reverse=True
    )
    print('\nModel ranking:')

    for name, score in sorted_results:
        print(f'{name}: {score:.4f}')

    print(f'\nBest model: {sorted_results[0][0]}')


if __name__ == '__main__':
    compare_models()
