# CSC4007 — Lab 1: Data Auditing & Data Card

Lab 1 giữ **IMDB/CineSense** là case study chính và bổ sung một **Vietnamese Transfer Challenge** để sinh viên kiểm tra xem tư duy data-centric có chuyển nguyên sang dữ liệu tiếng Việt hay không.

## 1. Mục tiêu

Sinh viên cần:

- audit schema, missingness, length distribution, duplicates/near-duplicates và leakage;
- dùng Great Expectations để kiểm tra các expectation cơ bản;
- dùng Cleanlab để rà soát label issues khi dataset có nhãn phù hợp;
- viết Data Card cho IMDB;
- thực hiện một transfer challenge ngắn trên dữ liệu tiếng Việt và so sánh với IMDB.

## 2. Cài đặt

```bash
conda activate csc4007-nlp
pip install -U typing_extensions
```

## 3. Part A — IMDB/CineSense (bắt buộc)

Chạy pipeline đầy đủ như phiên bản Lab 1 trước đây:

```bash
python run_lab1.py \
  --dataset imdb \
  --seed 42 \
  --use_great_expectations \
  --use_cleanlab
```

Output chính vẫn nằm ở `outputs/` để tương thích với tài liệu Lab hiện tại:

```text
outputs/
  logs/
  ge/
  splits/
  datacard_stats.json
```

Sinh viên vẫn phải review tối thiểu 5 mẫu trong `cleanlab_label_issues.csv` và hoàn thiện Data Card IMDB.

## 4. Part B — Vietnamese Transfer Challenge

### 4.1 Dataset tiếng Việt

Repo **không hard-code schema VietNewsSense**. Giảng viên có thể cung cấp một file CSV/TSV/JSON/JSONL/Parquet và chỉ định cột tương ứng.

Ví dụ nếu file có các cột `article_id`, `content`, `category`, `source`, `published_at`:

```bash
python run_lab1.py \
  --dataset vietnewssense \
  --data_path data/raw/vietnewssense_sample.csv \
  --id_column article_id \
  --text_column content \
  --label_column category \
  --seed 42 \
  --use_great_expectations
```

Nếu dataset chưa có nhãn, chỉ cần bỏ `--label_column` bằng cách dùng một tên cột không tồn tại hoặc chạy với file đã chuẩn hóa có cột `label` rỗng. Khi không có nhãn, pipeline vẫn audit được schema, Unicode, độ dài, duplicate và metadata; Cleanlab sẽ không chạy.

### 4.2 Output tiếng Việt

Để không ghi đè bằng chứng IMDB, output được tách riêng:

```text
outputs/vietnewssense/
  logs/audit_before.md
  logs/audit_after.md
  ge/
  splits/
  datacard_stats.json
```

### 4.3 Những gì mới được audit cho dữ liệu tiếng Việt

Ngoài checklist IMDB, transfer mode bổ sung:

- Unicode NFC normalization;
- zero-width / control characters;
- replacement characters do lỗi encoding;
- metadata như source/date/topic nếu file có;
- gợi ý phân tích source leakage, temporal leakage và event leakage.

Pipeline cố ý **không** thực hiện Vietnamese word segmentation, stop-word removal hay lowercase mạnh tay. Những lựa chọn đó thuộc Bài 3 về preprocessing.

## 5. Sản phẩm nộp

### Part A — IMDB

- `data_card.md`
- `datacard/heuristics_scorecard.md`
- `datacard/metadata_register.md`
- các file audit/GE/Cleanlab/splits trong `outputs/`

### Part B — Vietnamese Transfer Challenge

Sinh viên hoàn thiện ba file:

- `reports/vietnews_audit.md`
- `reports/transfer_matrix.md`
- `datacard/vietnews_mini_datacard.md`

Template cho ba sản phẩm này có sẵn trong repo.

## 6. Lưu ý về Cleanlab

Cleanlab chỉ có ý nghĩa khi:

- dataset có label rõ ràng;
- mỗi lớp có đủ mẫu;
- task/label guideline đã được định nghĩa;
- một baseline classifier hợp lý có thể được xây dựng.

Vì vậy, **Cleanlab bắt buộc ở IMDB nhưng không mặc định bắt buộc ở VietNewsSense**.

Nếu muốn thử Cleanlab với dataset tiếng Việt có nhãn:

```bash
python run_lab1.py \
  --dataset vietnewssense \
  --data_path data/raw/vietnewssense_sample.csv \
  --text_column content \
  --label_column category \
  --use_cleanlab
```

## 7. Câu hỏi transfer quan trọng

Sinh viên không chỉ báo cáo con số. Với mỗi vấn đề, cần trả lời:

1. Bằng chứng số liệu là gì?
2. Vì sao vấn đề này nguy hiểm?
3. Checklist từ IMDB có giữ nguyên được không?
4. Nếu phải thay đổi, thay đổi vì đặc điểm nào của dữ liệu tiếng Việt/tin tức?
