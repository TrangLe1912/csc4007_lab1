from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split


def _safe_stratify(series: pd.Series):
    if series.isna().any() or series.nunique(dropna=True) < 2:
        return None
    counts = series.value_counts()
    if counts.empty or int(counts.min()) < 2:
        return None
    return series


def make_splits(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    if len(df) < 3:
        return {
            "train": df.reset_index(drop=True),
            "val": df.iloc[0:0].copy().reset_index(drop=True),
            "test": df.iloc[0:0].copy().reset_index(drop=True),
        }

    y = df["label"] if "label" in df.columns else pd.Series(pd.NA, index=df.index)
    strat = _safe_stratify(y)

    try:
        df_train, df_temp = train_test_split(
            df,
            test_size=(1 - train_ratio),
            random_state=seed,
            stratify=strat,
        )
    except ValueError:
        df_train, df_temp = train_test_split(
            df,
            test_size=(1 - train_ratio),
            random_state=seed,
            stratify=None,
        )

    if len(df_temp) < 2:
        return {
            "train": df_train.reset_index(drop=True),
            "val": df_temp.reset_index(drop=True),
            "test": df.iloc[0:0].copy().reset_index(drop=True),
        }

    temp_ratio = val_ratio + test_ratio
    val_share = val_ratio / temp_ratio
    strat_temp = _safe_stratify(df_temp["label"]) if "label" in df_temp.columns else None

    try:
        df_val, df_test = train_test_split(
            df_temp,
            test_size=(1 - val_share),
            random_state=seed,
            stratify=strat_temp,
        )
    except ValueError:
        df_val, df_test = train_test_split(
            df_temp,
            test_size=(1 - val_share),
            random_state=seed,
            stratify=None,
        )

    return {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }
