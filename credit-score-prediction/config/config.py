import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = os.environ.get(
    "DATA_PATH",
    str(BASE_DIR / "data" / "data_B_clean.csv")
)

TARGET = "Credit_Score"

RANDOM_STATE = 42
TEST_SIZE = 0.2

ARTIFACT_DIR = Path(
    os.environ.get("ARTIFACT_DIR", str(BASE_DIR / "artifacts"))
)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{BASE_DIR}/mlflow.db"
)

MLFLOW_EXPERIMENT = "Credit Score Prediction"
