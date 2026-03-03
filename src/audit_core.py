from __future__ import annotations
import re
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

@dataclass
class AuditConfig:
    seed: int = 42
    near_dup_sample_n: int = 1500
    near_dup_threshold: float = 0.92
    tfidf_max_features: int = 20000
    tfidf_ngram_range: tuple = (1, 2)

def audit_schema_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    return {
        "n_rows": int(len(df)),
        "missing_text_count": int(df["text"].isna().sum()),
        "empty_text_count": int(texts.str.strip().eq("").sum()),
        "missing_label_count": int(df["label"].isna().sum()),
        "label_counts": df["label"].value_counts(dropna=False).to_dict(),
    }

def audit_html_artifacts(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    return {
        "contains_br_tag_count": int(texts.str.contains(BR_RE).sum()),
        "contains_any_html_tag_count": int(texts.str.contains(HTML_TAG_RE).sum()),
        "contains_html_entity_count": int(texts.str.contains(ENTITY_RE).sum()),
    }

def audit_distribution_length(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    lens = texts.map(len).to_numpy()
    vc = df["label"].value_counts(dropna=False)
    return {
        "imbalance_ratio_max_over_min": float(vc.max()/max(vc.min(),1)) if len(vc) else 0.0,
        "len_chars_min": int(lens.min()) if len(lens) else 0,
        "len_chars_median": int(np.median(lens)) if len(lens) else 0,
        "len_chars_p95": int(np.percentile(lens,95)) if len(lens) else 0,
        "len_chars_max": int(lens.max()) if len(lens) else 0,
    }

def audit_duplicates(df: pd.DataFrame, cfg: AuditConfig) -> Dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    h = texts.map(sha1_text)
    dup_mask = h.duplicated(keep=False)
    exact_dup_count = int(dup_mask.sum())
    exact_dup_ratio = float(exact_dup_count/len(df)) if len(df) else 0.0

    n = len(df)
    sample_n = min(cfg.near_dup_sample_n, n)
    sample = df.assign(text_hash=h).sample(sample_n, random_state=cfg.seed).reset_index(drop=True)

    vec = TfidfVectorizer(max_features=cfg.tfidf_max_features, ngram_range=cfg.tfidf_ngram_range, min_df=2)
    X = vec.fit_transform(sample["text"].fillna("").astype(str))
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
    y = df["label"]
    strat = y if y.nunique() > 1 else None
    train_idx, _ = train_test_split(df.index, test_size=0.2, random_state=cfg.seed, stratify=strat)

    vec_bad = TfidfVectorizer(max_features=cfg.tfidf_max_features, ngram_range=cfg.tfidf_ngram_range, min_df=2)
    vec_good = TfidfVectorizer(max_features=cfg.tfidf_max_features, ngram_range=cfg.tfidf_ngram_range, min_df=2)

    vec_bad.fit(texts)
    vec_good.fit(texts.loc[train_idx])

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
