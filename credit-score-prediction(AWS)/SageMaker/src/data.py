import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Path data: bisa di-override via env variable untuk fleksibilitas di SageMaker
DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "data_B_clean.csv"),
)

TARGET = "Credit_Score"

CLASS_NAMES = ["Poor", "Standard", "Good"]

FEATURE_NAMES = [
    "Month",
    "Age",
    "Occupation",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance",
]

NUMERIC_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
]

CATEGORICAL_FEATURES = [
    "Month", "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
]


def load_dataset() -> pd.DataFrame:
    path = os.path.abspath(DATA_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data tidak ditemukan di: {path}\n"
            f"Pastikan data_B_clean.csv ada di folder data/, "
            f"atau set env variable DATA_PATH."
        )
    df = pd.read_csv(path)
    # Pastikan hanya kolom yang dibutuhkan
    available = [c for c in FEATURE_NAMES if c in df.columns]
    return df[available + [TARGET]]


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURE_NAMES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
