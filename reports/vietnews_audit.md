# Vietnamese Transfer Challenge — UVN-1 Audit Report

> Hoàn thiện file này sau khi chạy `python run_lab1.py --dataset uvn1 ...`.

## 1. Dataset snapshot

- Dataset: `undertheseanlp/UVN-1`
- Số mẫu:
- Số category:
- Số source:
- Khoảng `publish_date`:
- Tỷ lệ thiếu ở `title`, `content`, `category`, `source`, `publish_date`:

## 2. Ba vấn đề dữ liệu có bằng chứng

### Vấn đề 1
- **Bằng chứng số liệu:**
- **Vì sao nguy hiểm:**
- **Hướng xử lý:**
- **So với IMDB:** giữ nguyên hay điều chỉnh checklist? Vì sao?

### Vấn đề 2
- **Bằng chứng số liệu:**
- **Vì sao nguy hiểm:**
- **Hướng xử lý:**
- **So với IMDB:** giữ nguyên hay điều chỉnh checklist? Vì sao?

### Vấn đề 3
- **Bằng chứng số liệu:**
- **Vì sao nguy hiểm:**
- **Hướng xử lý:**
- **So với IMDB:** giữ nguyên hay điều chỉnh checklist? Vì sao?

## 3. UVN-1 specific observations

Kiểm tra ít nhất ba nhóm sau:

- Unicode normalization / encoding;
- phân phối `category`;
- phân phối `source`;
- exact/near duplicates;
- temporal leakage từ `publish_date`;
- event overlap;
- khả năng source bias/source leakage;
- sự nhất quán giữa documentation và dữ liệu thật.

## 4. Kết luận transfer

Viết 3–5 câu trả lời:

**“Tư duy data-centric từ IMDB chuyển sang UVN-1 được giữ nguyên ở đâu, và phải thích nghi ở đâu?”**
