import pandas as pd
from typing import Optional, Tuple

def preprocess_data(
    path: str,
    training: bool = True,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    # Load dataset
    df = pd.read_csv(path)

    # Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty")

    # Default target column for your dataset
    if training:
        if target_col is None:
            target_col = "salary"   # specific to your dataset

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        return X, y

    else:
        # For inference (no target column)
        return df, None