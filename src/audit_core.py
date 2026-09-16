from __future__ import annotations
import re
import unicodedata
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import sha1_text

BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;")
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")


@dataclass
class AuditConfig:
    seed: int = 42
    near_dup_sample_n: int = 1500
    near_dup_threshold: float = 0.92
    tfidf_max_features: int = 20000
    tfidf_ngram_range: tuple = (1, 2)


def _label_counts(df: pd.DataFrame) -> Dict[str, int]:
    if "label" not in df.columns:
        return {}
    counts = df["label"].value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def audit_schema_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    missing_by_column = {
        str(c): int(v)
        for c, v in df.isna().sum().items()
        if int(v) > 0
    }
    return {
        "n_rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "missing_text_count": int(df["text"].isna().sum()),
        "empty_text_count": int(texts.str.strip().eq("").sum()),
        "missing_label_count": int(df["label"].isna().sum()) if "label" in df.columns else int(len(df)),
        "label_counts": _label_counts(df),
        "missing_by_column_nonzero": missing_by_column,
    }


def audit_html_artifacts(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    return {
        "contains_br_tag_count": int(texts.str.contains(BR_RE).sum()),
        "contains_any_html_tag_count": int(texts.str.contains(HTML_TAG_RE).sum()),
        "contains_html_entity_count": int(texts.str.contains(ENTITY_RE).sum()),
    }


def audit_unicode_artifacts(df: pd.DataFrame) -> Dict[str, Any]:
    """Audit Unicode issues that are especially relevant for Vietnamese text."""
    texts = df["text"].fillna("").astype(str)

    non_nfc = 0
    zero_width = 0
    replacement_char = 0
    control_chars = 0

    for text in texts:
        if text != unicodedata.normalize("NFC", text):
            non_nfc += 1
        if ZERO_WIDTH_RE.search(text):
            zero_width += 1
        if "\ufffd" in text:
            replacement_char += 1
        if any(
            unicodedata.category(ch).startswith("C") and ch not in "\n\r\t"
            for ch in text
        ):
            control_chars += 1

    return {
        "non_nfc_text_count": int(non_nfc),
        "contains_zero_width_char_count": int(zero_width),
        "contains_replacement_char_count": int(replacement_char),
        "contains_control_char_count": int(control_chars),
        "note": "Với tiếng Việt, cần kiểm tra Unicode normalization trước tokenization/modeling.",
    }


def audit_metadata_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize extra metadata such as source/date/topic when supplied."""
    standard = {"id", "text", "label", "split_orig"}
    extra_columns = [c for c in df.columns if c not in standard]
    if not extra_columns:
        return {
            "extra_columns": [],
            "metadata_note": "Không có metadata bổ sung trong file đầu vào.",
        }

    missing = {str(c): int(df[c].isna().sum()) for c in extra_columns}
    unique = {str(c): int(df[c].nunique(dropna=True)) for c in extra_columns}
    return {
        "extra_columns": [str(c) for c in extra_columns],
        "missing_by_column": missing,
        "unique_values_by_column": unique,
        "metadata_note": (
            "Nếu có source/time/topic/event metadata, dùng chúng để thảo luận "
            "source leakage, temporal leakage và event leakage."
        ),
    }


def audit_distribution_length(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    lens = texts.map(len).to_numpy()

    labels = df["label"].dropna() if "label" in df.columns else pd.Series(dtype=object)
    vc = labels.value_counts()
    imbalance = None
    if len(vc) >= 2:
        imbalance = float(vc.max() / max(vc.min(), 1))

    return {
        "imbalance_ratio_max_over_min": imbalance,
        "len_chars_min": int(lens.min()) if len(lens) else 0,
        "len_chars_median": int(np.median(lens)) if len(lens) else 0,
        "len_chars_p95": int(np.percentile(lens, 95)) if len(lens) else 0,
        "len_chars_max": int(lens.max()) if len(lens) else 0,
    }


def audit_duplicates(df: pd.DataFrame, cfg: AuditConfig) -> Dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    h = texts.map(sha1_text)
    dup_mask = h.duplicated(keep=False)
    exact_dup_count = int(dup_mask.sum())
    exact_dup_ratio = float(exact_dup_count / len(df)) if len(df) else 0.0

    n = len(df)
    sample_n = min(cfg.near_dup_sample_n, n)
    if sample_n < 2:
        return {
            "exact_dup_count": exact_dup_count,
            "exact_dup_ratio": exact_dup_ratio,
            "near_dup_pairs_found_in_sample": 0,
            "near_dup_note": "Không đủ dòng để kiểm tra near-duplicate.",
        }

    sample = (
        df.assign(text_hash=h)
        .sample(sample_n, random_state=cfg.seed)
        .reset_index(drop=True)
    )

    vec = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        ngram_range=cfg.tfidf_ngram_range,
        min_df=1 if sample_n < 20 else 2,
    )
    try:
        X = vec.fit_transform(sample["text"].fillna("").astype(str))
    except ValueError as exc:
        return {
            "exact_dup_count": exact_dup_count,
            "exact_dup_ratio": exact_dup_ratio,
            "near_dup_pairs_found_in_sample": 0,
            "near_dup_note": f"Bỏ qua TF-IDF near-duplicate: {exc}",
        }

    sim = cosine_similarity(X, dense_output=False)
    rows, cols = sim.nonzero()
    pairs = 0
    for i, j in zip(rows, cols):
        if i >= j:
            continue
        if float(sim[i, j]) >= cfg.near_dup_threshold:
            pairs += 1
            if pairs >= 30:
                break

    return {
        "exact_dup_count": exact_dup_count,
        "exact_dup_ratio": exact_dup_ratio,
        "near_dup_pairs_found_in_sample": pairs,
    }


def leakage_demo_tfidf(df: pd.DataFrame, cfg: AuditConfig) -> Dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    if len(df) < 3 or texts.str.strip().eq("").all():
        return {
            "status": "skipped",
            "reason": "Không đủ dữ liệu dùng được cho leakage demo.",
            "fix": "Split first. Fit preprocessing on train only; transform val/test.",
        }

    labels = df["label"] if "label" in df.columns else pd.Series(pd.NA, index=df.index)
    strat = None
    if labels.notna().all() and labels.nunique() > 1:
        min_class_count = int(labels.value_counts().min())
        if min_class_count >= 2:
            strat = labels

    try:
        train_idx, _ = train_test_split(
            df.index,
            test_size=0.2,
            random_state=cfg.seed,
            stratify=strat,
        )
    except ValueError:
        train_idx, _ = train_test_split(
            df.index,
            test_size=0.2,
            random_state=cfg.seed,
            stratify=None,
        )

    min_df = 1 if len(df) < 20 else 2
    vec_bad = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        ngram_range=cfg.tfidf_ngram_range,
        min_df=min_df,
    )
    vec_good = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        ngram_range=cfg.tfidf_ngram_range,
        min_df=min_df,
    )

    try:
        vec_bad.fit(texts)
        vec_good.fit(texts.loc[train_idx])
    except ValueError as exc:
        return {
            "status": "skipped",
            "reason": f"TF-IDF không fit được: {exc}",
            "fix": "Split first. Fit preprocessing on train only; transform val/test.",
        }

    return {
        "vocab_size_bad_fit_all": int(len(vec_bad.vocabulary_)),
        "vocab_size_good_fit_train": int(len(vec_good.vocabulary_)),
        "fix": "Split first. Fit preprocessing on train only; transform val/test.",
    }


def render_audit_md(path: str, title: str, sections: List[tuple[str, Dict[str, Any]]]) -> None:
    lines = [f"# {title}\n\n"]
    for sec_title, obj in sections:
        lines.append(f"## {sec_title}\n")
        for k, v in obj.items():
            lines.append(f"- **{k}**: {v}\n")
        lines.append("\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
