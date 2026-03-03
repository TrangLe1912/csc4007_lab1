from __future__ import annotations
import pandas as pd
from datasets import load_dataset

def load_imdb(max_rows: int | None = None) -> pd.DataFrame:
    ds = load_dataset("imdb")
    df_train = pd.DataFrame(ds["train"])
    df_test = pd.DataFrame(ds["test"])
    df_train["split_orig"] = "train"
    df_test["split_orig"] = "test"
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["id"] = range(len(df))
    if max_rows is not None:
        df = df.head(int(max_rows)).copy()
    return df[["id", "text", "label", "split_orig"]]

def load_dataset_any(name: str, max_rows: int | None = None) -> pd.DataFrame:
    if name == "imdb":
        return load_imdb(max_rows=max_rows)
    raise ValueError(f"Unsupported dataset: {name}")
