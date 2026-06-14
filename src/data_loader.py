import pandas as pd
from src import config


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    df = pd.read_csv(config.DATA_PATH)
    return df


def split_features_target(df: pd.DataFrame):
    """
    Split the dataframe into:
    - X: feature columns (drop target, ID column, and known leaked columns)
    - y: target column (F3924)
    """
    y = df[config.TARGET_COL]

    drop_cols = [config.TARGET_COL]
    if config.ID_COL in df.columns:
        drop_cols.append(config.ID_COL)
    drop_cols += [c for c in config.LEAKED_COLUMNS if c in df.columns]

    X = df.drop(columns=drop_cols)
    return X, y


def load_dataset():
    """Convenience function: load raw data and return (X, y)."""
    df = load_raw_data()
    X, y = split_features_target(df)
    return X, y