import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src import config


def get_feature_types(X: pd.DataFrame):
    """Split feature names into numeric and categorical lists."""
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X.columns.difference(cat_cols).tolist()
    return num_cols, cat_cols


def build_pipeline(num_cols, cat_cols) -> ColumnTransformer:
    """Build the preprocessing pipeline (impute + scale/encode)."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])

    return preprocessor


def fit_transform_features(X: pd.DataFrame, selected_features: list):
    """
    Given the full feature dataframe and the list of selected features,
    fit the preprocessing pipeline and return the transformed array
    plus the fitted pipeline (for later reuse) and output feature names.
    """
    X_sel = X[selected_features]
    num_cols, cat_cols = get_feature_types(X_sel)

    pipeline = build_pipeline(num_cols, cat_cols)
    X_processed = pipeline.fit_transform(X_sel)

    # Get output feature names (numeric stay same, categorical get one-hot expanded)
    cat_feature_names = []
    if cat_cols:
        encoder = pipeline.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()

    feature_names = num_cols + cat_feature_names

    return X_processed, pipeline, feature_names


def save_pipeline(pipeline, path=None):
    path = path or (config.MODELS_DIR / "preprocessor.pkl")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Saved preprocessing pipeline to {path}")