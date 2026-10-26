from pathlib import Path

import joblib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CHARTS_DIR = Path("charts")

def show(chart_name: str) -> None:
    CHARTS_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{chart_name}.png", dpi=120)
    plt.show()
    plt.close("all")

def show_df_correlation(X: pd.DataFrame, y: pd.Series, categorical_features: list[str]) -> None:
    X_encoded = pd.get_dummies(X, columns=categorical_features, dtype=float)
    feature_corr = X_encoded.corr(method="pearson")
    print("--- Feature-to-Feature Correlation Matrix ---")
    print(feature_corr.iloc[:5, :5].to_string())

    label_corr = X_encoded.corrwith(y, method="pearson").sort_values(ascending=False)
    print("\n--- Feature-to-Label Correlation ---")
    print(label_corr.to_string())


def correlation_heat_map(df: pd.DataFrame, categorical_features: list[str] = None) -> None:
    if categorical_features:
        df = pd.get_dummies(df, columns=categorical_features, dtype=float)
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Feature-to-Feature Correlation Heatmap")
    plt.tight_layout()
    show("df_correlation_heat_map")


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, categorical_features),
    ])
    return preprocessor


def fit_models(models: dict[str, Pipeline], X, y) -> None:
    for name, pipeline in models.items():
        pipeline.fit(X, y)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")


def save_model(model: Pipeline, model_path: Path, features: list[str], features_name_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    features_name_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    pd.Series(features, name="feature").to_csv(features_name_path, index=False)

