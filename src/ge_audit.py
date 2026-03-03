from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import great_expectations as ge

def run_great_expectations(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ge_df = ge.from_pandas(df)

    ge_df.expect_column_values_to_not_be_null("text")
    ge_df.expect_column_values_to_not_be_null("label")
    ge_df.expect_column_values_to_be_in_set("label", [0, 1])
    ge_df.expect_column_values_to_not_be_null("id")
    ge_df.expect_column_values_to_be_unique("id")

    ge_df["len_chars"] = ge_df["text"].fillna("").astype(str).str.len()
    ge_df.expect_column_values_to_be_between("len_chars", min_value=1, max_value=10000)

    validation = ge_df.validate(result_format="SUMMARY")
    suite = ge_df.get_expectation_suite(discard_failed_expectations=False)

    (out_dir / "expectation_suite.json").write_text(json.dumps(suite.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "validation_result.json").write_text(json.dumps(validation.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    stats = validation["statistics"]
    (out_dir / "validation_summary.md").write_text(
        "\n".join([
            "# Great Expectations — Validation Summary",
            f"- evaluated_expectations: {stats.get('evaluated_expectations')}",
            f"- successful_expectations: {stats.get('successful_expectations')}",
            f"- unsuccessful_expectations: {stats.get('unsuccessful_expectations')}",
            f"- success_percent: {stats.get('success_percent')}",
            "",
            "Nếu FAIL: ghi nguyên nhân và kế hoạch xử lý trong Data Card (Validation Types)."
        ]) + "\n",
        encoding="utf-8"
    )
    return {"ge_success": bool(validation["success"]), "ge_stats": stats}
