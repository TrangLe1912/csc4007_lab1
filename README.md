# CSC4007 — Lab 1: Data Auditing & Data Card

Lab 1 gồm hai phần:
- Part A: IMDB/CineSense — audit đầy đủ, Great Expectations, Cleanlab và Data Card.
- Part B: UVN-1 — Vietnamese Transfer Challenge.

## Dataset tiếng Việt

Sử dụng `undertheseanlp/UVN-1` từ Hugging Face. Code tự tải dữ liệu khi chạy.

Mapping:
- `content -> text`
- `category -> label`
- giữ metadata: `id`, `title`, `source`, `url`, `publish_date`

## Chạy Part A

```bash
python run_lab1.py --dataset imdb --seed 42 --use_great_expectations --use_cleanlab
```

## Chạy Part B

```bash
python run_lab1.py --dataset uvn1 --seed 42 --use_great_expectations
```

Output UVN-1 nằm tại `outputs/uvn1/` để không ghi đè minh chứng IMDB.

## Sản phẩm nộp

IMDB:
- `data_card.md`
- `datacard/heuristics_scorecard.md`
- `datacard/metadata_register.md`

UVN-1:
- `reports/vietnews_audit.md`
- `reports/transfer_matrix.md`
- `datacard/vietnews_mini_datacard.md`

Không commit toàn bộ `outputs/`; chỉ commit báo cáo và Data Card.

## Trọng tâm audit UVN-1

- schema và missingness
- Unicode/encoding
- class distribution
- exact/near duplicates
- source bias/source leakage
- temporal leakage dựa trên `publish_date`
- event overlap giữa các split
- đối chiếu documentation với dữ liệu thực tế

Lab 1 chưa làm Vietnamese word segmentation; phần đó để bài preprocessing.

## GitHub Actions

CI kiểm tra cấu trúc repo, cú pháp Python và smoke test trên tập nhỏ cho cả IMDB và UVN-1. Chất lượng phân tích được giảng viên chấm riêng theo rubric.
