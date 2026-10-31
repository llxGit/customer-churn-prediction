from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES 


def build_preprocessor() -> ColumnTransformer:
    numerical_features = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    categorical_features = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_features, NUMERICAL_FEATURES),
            ('cat', categorical_features, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor
 