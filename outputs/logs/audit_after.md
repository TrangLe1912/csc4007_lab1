# Audit AFTER preprocessing

## Schema / Missingness
- **n_rows**: 50000
- **columns**: ['id', 'text', 'label', 'split_orig']
- **missing_text_count**: 0
- **empty_text_count**: 0
- **missing_label_count**: 0
- **n_labels**: 2
- **label_counts**: {0: 25000, 1: 25000}

## HTML tags / Entities
- **contains_br_tag_count**: 0
- **contains_any_html_tag_count**: 0
- **contains_html_entity_count**: 11
- **example_snippets_with_html**: []

## Distribution / Length
- **label_counts**: {0: 25000, 1: 25000}
- **imbalance_ratio_max_over_min**: 1.0
- **len_chars_min**: 32
- **len_chars_median**: 954
- **len_chars_p95**: 3328
- **len_chars_max**: 13593
- **median_len_by_label**: {'0': 957.0, '1': 952.0}
- **note**: Length bias is a risk signal; do not conclude shortcut without further tests.

## Duplicates / Near-duplicates
- **exact_dup_count**: 832
- **exact_dup_ratio**: 0.01664
- **label_inconsistent_dup_groups**: 0
- **near_dup_pairs_found_in_sample**: 0
- **near_dup_examples**: []

## Leakage demo (TF-IDF fit all vs train-only)
- **demo_note**: Educational demo: fitting preprocessing on all data before split causes contamination.
- **vocab_size_bad_fit_all**: 50000
- **vocab_size_good_fit_train**: 50000
- **idf_bad_mean**: 7.311020672728293
- **idf_good_mean**: 7.3067675732285995
- **fix**: Split first. Fit preprocessing on train only; transform val/test.

