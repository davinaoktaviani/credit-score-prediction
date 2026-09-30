import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data import NUMERIC_FEATURES, CATEGORICAL_FEATURES

_sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
_ohe_sparse_kwarg = (
    {"sparse_output": False} if _sklearn_version >= (1, 2) else {"sparse": False}
)


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", **_ohe_sparse_kwarg)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe,    CATEGORICAL_FEATURES),
    ])


def build_pipelines() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("preprocessor", _build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "DecisionTree": Pipeline([
            ("preprocessor", _build_preprocessor()),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("preprocessor", _build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=30, max_depth=12,
                min_samples_leaf=5, n_jobs=-1, random_state=42,
            )),
        ]),
    }
