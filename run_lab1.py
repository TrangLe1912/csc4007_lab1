from __future__ import annotations
import argparse
from pathlib import Path
import json
import datetime

from src.utils import set_seed
from src.load_data import load_dataset_any
from src.preprocess import basic_clean
from src.split import make_splits

from src.audit_core import (
    AuditConfig,
    audit_schema_missingness,
    audit_html_artifacts,
    audit_unicode_artifacts,
    audit_metadata_missingness,
    audit_distribution_length,
    audit_duplicates,
    leakage_demo_tfidf,
    render_audit_md,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imdb", choices=["imdb", "uvn1", "local_csv"])
    ap.add_argument("--data_path", default=None, help="Required only for local_csv.")
    ap.add_argument("--text_column", default="text")
    ap.add_argument("--label_column", default="label")
    ap.add_argument("--id_column", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--use_great_expectations", action="store_true")
    ap.add_argument("--use_cleanlab", action="store_true")
    return ap


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    # Preserve legacy IMDB output paths used by the current Lab 1 handout.
    # UVN-1 is isolated so the Vietnamese transfer challenge never overwrites Part A evidence.
    out_dir = Path("outputs") if args.dataset == "imdb" else Path("outputs") / args.dataset
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    df = load_dataset_any(
        args.dataset,
        max_rows=args.max_rows,
        data_path=args.data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        id_column=args.id_column,
    )
    df["text"] = df["text"].fillna("").astype(str)

    cfg = AuditConfig(seed=args.seed)
    has_labels = bool(df["label"].notna().any())

    sec_before = [
        ("Schema / Missingness", audit_schema_missingness(df)),
        ("HTML tags / Entities", audit_html_artifacts(df)),
        ("Unicode / Vietnamese text", audit_unicode_artifacts(df)),
        ("Distribution / Length", audit_distribution_length(df)),
        ("Duplicates / Near-duplicates", audit_duplicates(df, cfg)),
        ("Leakage demo (TF-IDF fit all vs train-only)", leakage_demo_tfidf(df, cfg)),
        ("Optional metadata", audit_metadata_missingness(df)),
    ]
    render_audit_md(
        str(out_dir / "logs" / "audit_before.md"),
        f"Audit BEFORE preprocessing — {args.dataset}",
        sec_before,
    )

    df_clean = df.copy()
    df_clean["text"] = df_clean["text"].map(basic_clean)

    sec_after = [
        ("Schema / Missingness", audit_schema_missingness(df_clean)),
        ("HTML tags / Entities", audit_html_artifacts(df_clean)),
        ("Unicode / Vietnamese text", audit_unicode_artifacts(df_clean)),
        ("Distribution / Length", audit_distribution_length(df_clean)),
        ("Duplicates / Near-duplicates", audit_duplicates(df_clean, cfg)),
        ("Leakage demo (TF-IDF fit all vs train-only)", leakage_demo_tfidf(df_clean, cfg)),
        ("Optional metadata", audit_metadata_missingness(df_clean)),
    ]
    render_audit_md(
        str(out_dir / "logs" / "audit_after.md"),
        f"Audit AFTER preprocessing — {args.dataset}",
        sec_after,
    )

    splits = make_splits(df_clean, seed=args.seed)
    for name, d in splits.items():
        d.to_csv(out_dir / "splits" / f"{name}.csv", index=False)

    ge_info = None
    if args.use_great_expectations:
        from src.ge_audit import run_great_expectations
        allowed_labels = sorted(df_clean["label"].dropna().unique().tolist(), key=lambda x: str(x))
        ge_info = run_great_expectations(
            df_clean[["id", "text", "label"]].copy(),
            out_dir / "ge",
            require_label=has_labels,
            allowed_labels=allowed_labels if has_labels else None,
        )

    cl_info = None
    if args.use_cleanlab:
        if not has_labels:
            print("SKIP Cleanlab: dataset has no non-null labels.")
        else:
            from src.cleanlab_audit import run_cleanlab
            cv = 3 if args.max_rows else 5
            cl_info = run_cleanlab(
                df_clean[["id", "text", "label"]].copy(),
                out_dir / "logs",
                seed=args.seed,
                cv=cv,
            )

    schema_stats = audit_schema_missingness(df_clean)
    dist_stats = audit_distribution_length(df_clean)
    dup_stats = audit_duplicates(df_clean, cfg)
    html_stats = audit_html_artifacts(df_clean)
    unicode_stats = audit_unicode_artifacts(df_clean)
    metadata_stats = audit_metadata_missingness(df_clean)

    stats = {
        "dataset": args.dataset,
        "source_path": args.data_path,
        "seed": args.seed,
        "max_rows": args.max_rows,
        "n_rows": schema_stats["n_rows"],
        "label_counts": schema_stats["label_counts"],
        "text_length": {k: dist_stats[k] for k in ["len_chars_min", "len_chars_median", "len_chars_p95", "len_chars_max"]},
        "html_artifacts": html_stats,
        "unicode_artifacts": unicode_stats,
        "metadata": metadata_stats,
        "duplicates": dup_stats,
        "splits": {k: int(len(v)) for k, v in splits.items()},
        "great_expectations": ge_info,
        "cleanlab": cl_info,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (out_dir / "datacard_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"DONE. Results: {out_dir}")


if __name__ == "__main__":
    main()
