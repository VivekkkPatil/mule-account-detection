import json
import time
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src import config


def drop_high_missing(X: pd.DataFrame) -> pd.DataFrame:
    missing_ratio = X.isnull().mean()
    keep_cols = missing_ratio[missing_ratio <= config.MAX_MISSING_RATIO].index
    print(f"Dropping {X.shape[1] - len(keep_cols)} columns "
          f"(> {config.MAX_MISSING_RATIO:.0%} missing)")
    return X[keep_cols]


def simple_impute_and_encode(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns
    num_cols = X.columns.difference(cat_cols)

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0])
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    return X


def drop_zero_variance(X: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are constant (zero variance) - quick, cheap win."""
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X)
    keep_cols = X.columns[selector.get_support()]
    print(f"Dropping {X.shape[1] - len(keep_cols)} zero-variance columns")
    return X[keep_cols]


def rank_by_mutual_info(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    print("Computing mutual information (this is the slow step)...")
    start = time.time()
    mi_scores = mutual_info_classif(
        X, y,
        random_state=config.RANDOM_STATE,
        discrete_features=False,  # treat all as continuous -> much faster
    )
    print(f"Mutual info done in {time.time() - start:.1f}s")
    mi_series = pd.Series(mi_scores, index=X.columns)
    return mi_series.rank(ascending=False)


def rank_by_xgb_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    print("Training XGBoost for importance ranking...")
    start = time.time()
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X, y)
    print(f"XGBoost done in {time.time() - start:.1f}s")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return importances.rank(ascending=False)


def select_top_features(X: pd.DataFrame, y: pd.Series) -> list:
    X_reduced = drop_high_missing(X)
    X_ranked = simple_impute_and_encode(X_reduced)
    X_ranked = drop_zero_variance(X_ranked)

    mi_rank = rank_by_mutual_info(X_ranked, y)
    xgb_rank = rank_by_xgb_importance(X_ranked, y)

    combined_rank = (mi_rank + xgb_rank) / 2
    top_features = combined_rank.sort_values().head(config.TOP_N_FEATURES).index.tolist()

    print(f"Selected top {len(top_features)} features (from {X.shape[1]} original).")
    return top_features


def save_selected_features(features: list):
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SELECTED_FEATURES_PATH, "w") as f:
        json.dump(features, f, indent=2)
    print(f"Saved selected features to {config.SELECTED_FEATURES_PATH}")


if __name__ == "__main__":
    from src.data_loader import load_dataset

    X, y = load_dataset()
    top_features = select_top_features(X, y)
    save_selected_features(top_features)