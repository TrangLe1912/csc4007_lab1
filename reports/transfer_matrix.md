# Transfer Matrix — IMDB ↔ Vietnamese News

| Audit question | IMDB/CineSense | Vietnamese news / VietNewsSense | Giữ nguyên hay điều chỉnh? | Bằng chứng từ output |
|---|---|---|---|---|
| Schema hợp lệ? | `review/text`, `sentiment/label` |  |  |  |
| Missing values? | text/label |  |  |  |
| Class imbalance? | gần cân bằng |  |  |  |
| Exact duplicate? | có thể có |  |  |  |
| Near-duplicate? | review gần giống |  |  |  |
| Leakage? | train/test contamination, movie-level risk |  |  |  |
| Label issues? | score ↔ text conflict |  |  |  |
| Language-specific issue? | HTML entities/tags |  |  |  |
| Metadata risk? | ít metadata |  |  |  |

## Kết luận ngắn

Viết 5–7 câu chỉ ra:

- kiểm tra nào chuyển nguyên từ IMDB sang dữ liệu tiếng Việt;
- kiểm tra nào phải thay đổi cách diễn giải;
- vấn đề mới nào chỉ nổi bật khi làm việc với tin tức tiếng Việt.
