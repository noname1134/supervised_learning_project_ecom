import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import utils

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.3

CHURN_COLUMN = "Churn"
BEST_MODEL_NAME = "random_forest"

NA_TOKENS = ["", "NA", "N/A", "n/a", "-", "missing", "null", "None"]
NUMERIC_FEATURES = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]
CATEGORICAL_FEATURES = ["Gender", "Subscription Type", "Contract Length"]

DATA_PATH = Path("data/customer_churn_dataset.csv")

MODEL_PATH = Path("models/churn_model.joblib")
FEATURE_NAMES_PATH = Path("models/churn_model_features_names.csv")


def load_data(path: Path) -> pd.DataFrame:
    utils.require_file(path)
    df = pd.read_csv(path, na_values=NA_TOKENS)
    df_cleaned = df.drop_duplicates(subset=["CustomerID"], keep='first')
    df_cleaned.drop(columns=["CustomerID"], inplace=True)
    df_cleaned = df_cleaned.replace(r"^\s*$", np.nan, regex=True)
    return df_cleaned


def create_pipeline(classifier: BaseEstimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", utils.build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        ("classifier", classifier),
    ])


def get_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": create_pipeline(LogisticRegressionCV(cv=CV_FOLDS, max_iter=5000, random_state=RANDOM_STATE)),
        "knn": create_pipeline(
            GridSearchCV(
                KNeighborsClassifier(),
                param_grid={"n_neighbors": [3, 5, 7, 9, 11]},
                cv=CV_FOLDS,
                scoring="accuracy",
                n_jobs=-1,
            )
        ),
        "svm": create_pipeline(
            GridSearchCV(
                SVC(random_state=RANDOM_STATE),
                param_grid={"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
                cv=CV_FOLDS,
                scoring="accuracy",
                n_jobs=-1,
            )
        ),
        "random_forest": create_pipeline(
            GridSearchCV(
                RandomForestClassifier(random_state=RANDOM_STATE),
                param_grid={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "bootstrap": [True, False],
                },
                cv=CV_FOLDS,
                scoring="accuracy",
                n_jobs=-1,
            )
        ),
    }


def print_models_optimal_parameters(models: dict[str, Pipeline]) -> None:
    print("Optimal parameters per model")
    for name, pipeline in models.items():
        print(f"\nModel: {name}")
        classifier = pipeline.named_steps["classifier"]

        if name == "logistic_regression":
            if hasattr(classifier, "C_"):
                print(f"Optimal C: {classifier.C_}")
            else:
                print("Optimal C: Not available")

        elif hasattr(classifier, "best_params_"):
            for param, value in classifier.best_params_.items():
                print(f"{param}: {value}")
        else:
            print("No optimal parameters found for this classifier.")


def evaluate_models(models: dict[str, Pipeline], X_test, y_test) -> pd.DataFrame:
    evaluation_results = []
    for name, pipeline in models.items():
        predictions = pipeline.predict(X_test)
        print(f"model {name} predictions:")
        print(predictions)

        if name != "svm":
            probabilities = pipeline.predict_proba(X_test)
            print(f"model {name} prediction probabilities:")
            print(probabilities)

        cm = confusion_matrix(y_test, predictions)
        plot_confusion_matrices(cm, name)

        evaluation_results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions, zero_division=0)
        })
    eval_df = pd.DataFrame(evaluation_results)
    eval_df = eval_df.sort_values(by="accuracy").reset_index(drop=True)
    return eval_df


def plot_confusion_matrices(cm, model_name: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
    )

    plt.title(
        f"Confusion Matrix: {model_name.replace('_', ' ').title()}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)

    utils.show(f"{model_name}_confusion_matrix_heat_map")


def main() -> None:
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=[CHURN_COLUMN]), df[CHURN_COLUMN]
    utils.show_df_correlation(X, y, CATEGORICAL_FEATURES)
    utils.correlation_heat_map(df, CATEGORICAL_FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    models = get_models()
    utils.fit_models(models, X_train, y_train)
    print_models_optimal_parameters(models)

    eval_df = evaluate_models(models, X_test, y_test)
    print("Models evaluation dataframe")
    print(eval_df.to_string())

    models[BEST_MODEL_NAME].fit(X_train, y_train)
    optimal_pipeline = models[BEST_MODEL_NAME]
    optimal_pipeline.fit(X, y)
    utils.save_model(optimal_pipeline, MODEL_PATH, list(X.columns), FEATURE_NAMES_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved feature names -> {FEATURE_NAMES_PATH}")
    
    loaded_model = joblib.load(MODEL_PATH)



if __name__ == "__main__":
    main()