from __future__ import annotations
from pathlib import Path
import pandas as pd

UVN1_DATASET_ID = "undertheseanlp/UVN-1"


def load_imdb(max_rows: int | None = None) -> pd.DataFrame:
    from datasets import load_dataset
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


def load_uvn1(max_rows: int | None = None) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(UVN1_DATASET_ID, split="train")
    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))
    df = ds.to_pandas().copy()
    required = {"id", "content", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected UVN-1 schema. Missing: {sorted(missing)}; available: {list(df.columns)}")
    df["text"] = df["content"]
    df["label"] = df["category"]
    keep = ["id", "text", "label", "title", "source", "url", "publish_date"]
    return df[[c for c in keep if c in df.columns]]


def _read_local_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".tsv":
        return pd.read_csv(p, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if suffix == ".json":
        return pd.read_json(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported data format: {suffix}")


def load_local_dataset(data_path: str, *, text_column: str = "text", label_column: str | None = "label", id_column: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    df = _read_local_table(data_path)
    df = df.head(int(max_rows)).copy() if max_rows is not None else df.copy()
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available: {list(df.columns)}")
    result = df.copy()
    result["text"] = result[text_column]
    result["label"] = result[label_column] if label_column and label_column in result.columns else pd.NA
    if id_column:
        if id_column not in result.columns:
            raise ValueError(f"ID column '{id_column}' not found. Available: {list(result.columns)}")
        result["id"] = result[id_column]
    elif "id" not in result.columns:
        result["id"] = range(len(result))
    aliases_to_drop = {text_column}
    if label_column:
        aliases_to_drop.add(label_column)
    if id_column:
        aliases_to_drop.add(id_column)
    aliases_to_drop -= {"id", "text", "label"}
    result = result.drop(columns=[c for c in aliases_to_drop if c in result.columns])
    front = ["id", "text", "label"]
    remaining = [c for c in result.columns if c not in front]
    return result[front + remaining]


def load_dataset_any(name: str, max_rows: int | None = None, *, data_path: str | None = None, text_column: str = "text", label_column: str | None = "label", id_column: str | None = None) -> pd.DataFrame:
    if name == "imdb":
        return load_imdb(max_rows=max_rows)
    if name == "uvn1":
        return load_uvn1(max_rows=max_rows)
    if name == "local_csv":
        if not data_path:
            raise ValueError("--data_path is required when --dataset local_csv")
        return load_local_dataset(data_path, text_column=text_column, label_column=label_column, id_column=id_column, max_rows=max_rows)
    raise ValueError(f"Unsupported dataset: {name}")
