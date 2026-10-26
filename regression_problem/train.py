import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import utils

from pathlib import Path

from typing import Any
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoCV, Ridge, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

RANDOM_STATE = 42
TEST_SIZE = 0.3

TARGET_COLUMN = "GDP"
BEST_MODEL_NAME = "linear_regression"
ACCURACY_METRIX = ["MAE", "MSE", "RMSE"]

NA_TOKENS = ["", "NA", "N/A", "n/a", "-", "missing", "null", "None"]
NUMERIC_FEATURES = ["Date", "Confirmed", "Deaths", "Recovered", "Unemployment", "CPI"]
CATEGORICAL_FEATURES = ["Country/Region"]

DATA_PATH = Path("data/Covid19_With_GDP_Values.csv")

MODEL_PATH = Path("models/gdp_model.joblib")
FEATURE_NAMES_PATH = Path("models/gdp_model_features_names.csv")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=NA_TOKENS)
    df.drop(columns=["Province/State"], inplace=True)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df["Date"] = pd.to_datetime(df['Date']).dt.year
    df_cleaned = df.drop_duplicates(subset=['Country/Region', 'Date'], keep='first')
    return df_cleaned



def create_pipeline(estimator: BaseEstimator, include_poly: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [("preprocessor", utils.build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES))]
    if include_poly:
        steps.append(("poly", PolynomialFeatures(degree=1, include_bias=False)))

    steps.append(("estimator", estimator))
    return Pipeline(steps)


def get_models() -> dict[str, Pipeline]:
    return {
        "linear_regression": create_pipeline(LinearRegression()),
        "ridge_cv": create_pipeline(RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        "lasso_cv": create_pipeline(LassoCV(cv=5, max_iter=10000, random_state=RANDOM_STATE)),
        "polynomial_regression": create_pipeline(Ridge(alpha=10.0), include_poly=True)
    }


def print_models_info(models: dict[str, Pipeline]) -> None:
    print("Optimal parameters per model")

    for name, pipeline in models.items():
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        print(f"Model: {name}")
        if name == "linear_regression":
            print("Type: Vanilla OLS / Fixed Parameters")
            print(f"Intercept fit: {pipeline.named_steps['estimator'].fit_intercept}")

        elif name == "ridge_cv":
            best_alpha = pipeline.named_steps["estimator"].alpha_
            print("Type: Cross-Validated (RidgeCV)")
            print(f"Optimal Alpha (Lambda): {best_alpha}")

        elif name == "lasso_cv":
            best_alpha = pipeline.named_steps["estimator"].alpha_
            print("Type: Cross-Validated (LassoCV)")
            print(f"Optimal Alpha (Lambda): {best_alpha}")
            print(f"Number of iterations run: {pipeline.named_steps['estimator'].n_iter_}")

        elif name == "polynomial_regression":
            feature_names = pipeline.named_steps["poly"].get_feature_names_out(feature_names)
            degree = pipeline.named_steps["poly"].degree
            alpha_val = pipeline.named_steps["estimator"].alpha
            print("Type: Polynomial Pipeline + Ridge")
            print(f"Optimal Polynomial Degree: {degree}")
            print(f"Ridge Alpha (Lambda) Set: {alpha_val}")

        estimator = pipeline.named_steps["estimator"]
        coefficients = estimator.coef_
        intercept = estimator.intercept_

        coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
        coef_df["Abs_Magnitude"] = coef_df["Coefficient"].abs()
        coef_df = (coef_df.sort_values(by="Abs_Magnitude", ascending=False).drop(columns=["Abs_Magnitude"]).reset_index(drop=True))

        print(f"Model Intercept: {intercept:,.2f}")
        print("Feature Coefficients (sorted by magnitude):")
        print(coef_df.to_string(index=False))
        print("=" * 70, "\n")



def evaluate_models(models: dict[str, Pipeline], X_test, y_test) -> pd.DataFrame:
    evaluation_results = []

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        evaluation_results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        })

    eval_df = pd.DataFrame(evaluation_results)
    eval_df = eval_df.sort_values(by="RMSE").reset_index(drop=True)
    return eval_df


def show_models_accuracy(eval_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.set_theme(style="whitegrid")
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    for i, metric in enumerate(ACCURACY_METRIX):
        ax = axes[i]
        sorted_df = eval_df.sort_values(by=metric)
        metric_palette = [colors[i]] * len(sorted_df)

        sns.barplot(x="Model", y=metric, data=sorted_df, ax=ax, palette=metric_palette, hue="Model", legend=False)
        ax.set_title(f"Model Comparison by {metric}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Model Architecture", fontsize=11)
        ax.set_ylabel(f"{metric}", fontsize=11)
        ax.tick_params(axis="x", rotation=15)

    for p in ax.patches:
        height = p.get_height()
        label_text = f"{height:,.0e}" if height > 1e6 else f"{height:,.0f}"
        ax.annotate(label_text, (p.get_x() + p.get_width() / 2.0, height), ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

    plt.tight_layout()
    utils.show("models_accuracy")


def main() -> None:
    df = load_data(DATA_PATH)

    X, y = df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
    utils.show_df_correlation(X, y, CATEGORICAL_FEATURES)
    utils.correlation_heat_map(df[NUMERIC_FEATURES + [TARGET_COLUMN]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    models = get_models()
    utils.fit_models(models, X_train, y_train)
    print_models_info(models)

    eval_df = evaluate_models(models, X_test, y_test)
    show_models_accuracy(eval_df)

    optimal_pipeline = models[BEST_MODEL_NAME]
    optimal_pipeline.fit(X, y)
    utils.save_model(optimal_pipeline, MODEL_PATH, list(X.columns), FEATURE_NAMES_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved feature names -> {FEATURE_NAMES_PATH}")

    loaded_model = joblib.load(MODEL_PATH)


if __name__ == "__main__":
    main()