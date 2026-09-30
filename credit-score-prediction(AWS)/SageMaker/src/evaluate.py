from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


def evaluate_pipeline(
    pipeline: Pipeline, X_train, y_train, X_test, y_test
) -> dict:
    train_acc  = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc   = accuracy_score(y_test,  pipeline.predict(X_test))
    test_f1    = f1_score(y_test, pipeline.predict(X_test), average="weighted", zero_division=0)
    return {
        "train_accuracy":    train_acc,
        "test_accuracy":     test_acc,
        "test_f1_weighted":  test_f1,
    }


def print_comparison(results: dict[str, dict]) -> None:
    print(f"\n{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'Test F1':>10}")
    print("-" * 56)
    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m['train_accuracy']:>10.4f} "
            f"{m['test_accuracy']:>10.4f} "
            f"{m['test_f1_weighted']:>10.4f}"
        )


def print_classification_report(
    pipeline: Pipeline, X_test, y_test, target_names
) -> None:
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))


def select_best(results: dict[str, dict], metric: str = "test_f1_weighted") -> str:
    return max(results, key=lambda name: results[name][metric])
