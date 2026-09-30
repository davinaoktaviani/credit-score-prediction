import json
import os

import joblib
import numpy as np
import pandas as pd


JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE  = "text/csv"

CLASS_NAMES = ["Poor", "Standard", "Good"]

FEATURE_NAMES = [
    "Month", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
    "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour",
    "Monthly_Balance",
]


def model_fn(model_dir: str):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    if request_content_type == JSON_CONTENT_TYPE:
        payload   = json.loads(request_body)
        instances = payload["instances"]
        # instances bisa list of dict atau list of list
        if isinstance(instances[0], dict):
            return pd.DataFrame(instances, columns=FEATURE_NAMES)
        else:
            return pd.DataFrame(instances, columns=FEATURE_NAMES)

    raise ValueError(
        f"Unsupported content type: {request_content_type}. "
        f"Gunakan application/json dengan format: "
        f'{{"instances": [{{"Month": "April", "Age": 30, ...}}]}}'
    )


def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    probs     = pipeline.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels    = [pipeline.classes_[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions":   class_ids.tolist(),
        "labels":        labels,
    }


def output_fn(prediction: dict, accept_content_type: str):
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
