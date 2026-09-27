import joblib

from pathlib import Path


def save_artifact(
    obj,
    path: Path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        obj,
        path
    )


def load_artifact(path: Path):

    return joblib.load(path)