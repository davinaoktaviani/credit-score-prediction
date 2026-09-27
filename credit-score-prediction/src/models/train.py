import mlflow
import mlflow.sklearn

from config.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    ARTIFACT_DIR,
)

from src.utils.io import save_artifact


class Trainer:

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    def train(self, pipeline, X_train, y_train, run_name="model"):
        mlflow.end_run()

        with mlflow.start_run(run_name=run_name) as run:
            pipeline.fit(X_train, y_train)

            model = pipeline.named_steps["model"]

            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_params(model.get_params())

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            save_artifact(
                pipeline,
                ARTIFACT_DIR / f"{run_name}.pkl"
            )

            
            return pipeline, run.info.run_id
