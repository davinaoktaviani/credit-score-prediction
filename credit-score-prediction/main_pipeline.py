import mlflow
import mlflow.sklearn
import pandas as pd

from src.data.loader import DataLoader
from src.features.pipeline_preprocessor import PipelinePreprocessor
from src.pipelines.sklearn_pipeline import CreditScorePipeline
from src.models.train import Trainer
from src.models.evaluate import Evaluator

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    ARTIFACT_DIR,
)
from src.utils.io import save_artifact
import os


def main():
    loader = DataLoader()
    df = loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data(df)

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = PipelinePreprocessor(num_cols, cat_cols).build()

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=30,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
    }

    trainer  = Trainer()
    evaluator = Evaluator()

    results_list = []
    best_f1      = -1
    best_pipeline = None
    best_name     = None

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training model: {name}")
        print(f"{'='*60}")

        pipeline = CreditScorePipeline(preprocessor, model).build()

        preprocessor_fresh = PipelinePreprocessor(num_cols, cat_cols).build()
        pipeline = CreditScorePipeline(preprocessor_fresh, model).build()

        trained_model, run_id = trainer.train(
            pipeline, X_train, y_train, run_name=name
        )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_id=run_id):
            metrics = evaluator.evaluate(trained_model, X_test, y_test)

        metrics["model"]  = name
        metrics["run_id"] = run_id
        results_list.append(metrics)

        if metrics["f1_score"] > best_f1:
            best_f1       = metrics["f1_score"]
            best_pipeline = trained_model
            best_name     = name

    results_df = pd.DataFrame(results_list).sort_values(
        by="f1_score", ascending=False
    )

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print("\n")
    print("=" * 60)
    print(f"BEST MODEL: {best_name}  (F1={best_f1:.4f})")
    print("=" * 60)

    best_model_path = ARTIFACT_DIR / "best_model.pkl"
    save_artifact(best_pipeline, best_model_path)
    print(f"\nBest model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
