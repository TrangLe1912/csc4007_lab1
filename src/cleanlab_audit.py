from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from cleanlab.filter import find_label_issues

def run_cleanlab(df: pd.DataFrame, out_dir: Path, seed: int = 42, cv: int = 3) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    X = df["text"].fillna("").astype(str).values
    y = df["label"].values

    base_model = make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2),
        LogisticRegression(max_iter=200)
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    n_classes = len(np.unique(y))
    pred_probs = np.zeros((len(df), n_classes), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        model = clone(base_model)
        model.fit(X[train_idx], y[train_idx])
        pred_probs[test_idx] = model.predict_proba(X[test_idx])

    ranked_indices = list(find_label_issues(labels=y, pred_probs=pred_probs, return_indices_ranked_by="self_confidence"))
    issues_df = df.loc[ranked_indices].copy()
    issues_df["given_label_prob"] = pred_probs[ranked_indices, y[ranked_indices]]
    issues_df = issues_df.sort_values("given_label_prob", ascending=True)

    top_k = min(200, len(issues_df))
    issues_df.head(top_k)[["id","label","given_label_prob","text"]].to_csv(out_dir / "cleanlab_label_issues.csv", index=False)

    ratio = (len(ranked_indices) / max(len(df),1))
    (out_dir / "cleanlab_summary.md").write_text(
        "\n".join([
            "# Cleanlab — Label Issues Summary",
            f"- suspected_label_issues_count: {len(ranked_indices)}",
            f"- suspected_label_issues_ratio: {ratio:.4f}",
            f"- export_top_k: {top_k}",
            "",
            "Student task: chọn 5 mẫu trong cleanlab_label_issues.csv để review (giữ/sửa/ambiguous) và ghi vào Data Card."
        ]) + "\n",
        encoding="utf-8"
    )

    return {"suspected_count": len(ranked_indices), "suspected_ratio": ratio, "export_top_k": top_k}
