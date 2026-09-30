import os
import sys
import tarfile

import joblib
import mlflow
import mlflow.sklearn

sys.path.insert(0, "src")
from data import load_dataset, split_data, CLASS_NAMES
from models import build_pipelines
from evaluate import evaluate_pipeline, print_comparison, select_best, print_classification_report


ARTIFACT_DIR   = "model_artifact"
MODEL_FILENAME = "model.joblib"
TARBALL_PATH   = os.path.join(ARTIFACT_DIR, "model.tar.gz")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT   = "Credit Score - SageMaker Pipeline"


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
    print("(harus 1.2.x agar cocok dengan SageMaker container framework_version='1.2-1')\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")

    pipelines = build_pipelines()
    results   = {}

    for name, pipeline in pipelines.items():
        print(f"\n{'='*55}")
        print(f"Training: {name}")
        print(f"{'='*55}")

        mlflow.end_run()
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            metrics = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)

            mlflow.log_param("model_type", name)
            mlflow.log_metrics({
                "train_accuracy":    metrics["train_accuracy"],
                "test_accuracy":     metrics["test_accuracy"],
                "test_f1_weighted":  metrics["test_f1_weighted"],
            })
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        results[name] = metrics

    print_comparison(results)

    best_name     = select_best(results, metric="test_f1_weighted")
    best_pipeline = pipelines[best_name]

    best_pipeline.fit(X_train, y_train)

    print(f"\nWinner: {best_name}")
    print(f"\nDetailed report for {best_name}:")
    print_classification_report(best_pipeline, X_test, y_test, CLASS_NAMES)

    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    joblib.dump(best_pipeline, model_path, compress=3)
    print(f"\nSaved: {model_path}")

    with tarfile.open(TARBALL_PATH, "w:gz") as tar:
        tar.add(model_path, arcname=MODEL_FILENAME)
    print(f"Packaged: {TARBALL_PATH}")

    print("\n" + "="*55)
    print("NEXT STEPS")
    print("="*55)
    print("1. Buat S3 bucket LEWAT AWS CONSOLE (bukan CLI!):")
    print("   → AWS Console → S3 → Create bucket")
    print("   → Nama: credit-score-[namakamu]")
    print("   → Region: us-east-1\n")
    print("2. Upload model ke S3 via Console atau CLI:")
    print(f"   aws s3 cp {TARBALL_PATH} s3://NAMA-BUCKET/credit-score/model.tar.gz\n")
    print("   ATAU lewat Console: drag & drop model.tar.gz ke bucket\n")
    print("3. Deploy endpoint:")
    print("   python deploy_endpoint.py\n")


if __name__ == "__main__":
    main()
