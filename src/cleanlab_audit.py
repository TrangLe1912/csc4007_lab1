from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from cleanlab.filter import find_label_issues


def run_cleanlab(df: pd.DataFrame, out_dir: Path, seed: int = 42, cv: int = 3) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    work = df[df["label"].notna()].copy().reset_index(drop=True)
    if len(work) < 6:
        raise ValueError("Cleanlab cần ít nhất 6 mẫu có nhãn để chạy ổn định.")

    X = work["text"].fillna("").astype(str).values
    original_labels = work["label"].astype(str).values

    encoder = LabelEncoder()
    y = encoder.fit_transform(original_labels)
    n_classes = len(encoder.classes_)
    if n_classes < 2:
        raise ValueError("Cleanlab cần ít nhất 2 lớp nhãn.")

    min_class_count = int(pd.Series(y).value_counts().min())
    cv = max(2, min(int(cv), min_class_count))
    if min_class_count < 2:
        raise ValueError("Mỗi lớp cần ít nhất 2 mẫu để chạy cross-validation cho Cleanlab.")

    min_df = 1 if len(work) < 20 else 2
    base_model = make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=min_df),
        LogisticRegression(max_iter=300),
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    pred_probs = np.zeros((len(work), n_classes), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        model = clone(base_model)
        model.fit(X[train_idx], y[train_idx])
        fold_probs = model.predict_proba(X[test_idx])
        pred_probs[test_idx] = fold_probs

    ranked_indices = list(
        find_label_issues(
            labels=y,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
        )
    )

    issues_df = work.loc[ranked_indices].copy()
    encoded_given = y[ranked_indices]
    issues_df["given_label_prob"] = pred_probs[ranked_indices, encoded_given]
    issues_df = issues_df.sort_values("given_label_prob", ascending=True)

    top_k = min(200, len(issues_df))
    issues_df.head(top_k)[["id", "label", "given_label_prob", "text"]].to_csv(
        out_dir / "cleanlab_label_issues.csv",
        index=False,
    )

    ratio = len(ranked_indices) / max(len(work), 1)
    (out_dir / "cleanlab_summary.md").write_text(
        "\n".join(
            [
                "# Cleanlab — Label Issues Summary",
                f"- labeled_rows_used: {len(work)}",
                f"- classes: {list(encoder.classes_)}",
                f"- cv_folds: {cv}",
                f"- suspected_label_issues_count: {len(ranked_indices)}",
                f"- suspected_label_issues_ratio: {ratio:.4f}",
                f"- export_top_k: {top_k}",
                "",
                "Student task: chọn 5 mẫu trong cleanlab_label_issues.csv để review (giữ/sửa/ambiguous) và ghi vào Data Card.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "labeled_rows_used": len(work),
        "classes": list(encoder.classes_),
        "cv_folds": cv,
        "suspected_count": len(ranked_indices),
        "suspected_ratio": ratio,
        "export_top_k": top_k,
    }
