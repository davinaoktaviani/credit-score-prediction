import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


class Evaluator:

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)


        print(classification_report(y_test, y_pred, zero_division=0))

        return {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1,
        }
