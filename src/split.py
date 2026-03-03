from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

def make_splits(df: pd.DataFrame, seed: int, train_ratio: float=0.8, val_ratio: float=0.1, test_ratio: float=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    y = df["label"]
    strat = y if y.nunique() > 1 else None

    df_train, df_temp = train_test_split(df, test_size=(1-train_ratio), random_state=seed, stratify=strat)
    temp_ratio = val_ratio + test_ratio
    val_share = val_ratio / temp_ratio
    strat_temp = df_temp["label"] if df_temp["label"].nunique() > 1 else None

    df_val, df_test = train_test_split(df_temp, test_size=(1-val_share), random_state=seed, stratify=strat_temp)
    return {"train": df_train.reset_index(drop=True),
            "val": df_val.reset_index(drop=True),
            "test": df_test.reset_index(drop=True)}
