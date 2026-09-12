# Transfer Matrix — IMDB ↔ UVN-1

| Audit question | IMDB/CineSense | UVN-1 | Giữ nguyên hay điều chỉnh? | Bằng chứng từ output |
|---|---|---|---|---|
| Schema hợp lệ? | `text`, `label` | `content -> text`, `category -> label` + metadata |  |  |
| Missing values? | text/label | title/content/category/source/publish_date |  |  |
| Class imbalance? | gần cân bằng | theo `category` |  |  |
| Exact duplicate? | có thể có | kiểm tra cùng nội dung/khác URL |  |  |
| Near-duplicate? | review gần giống | bài đăng lại / cùng sự kiện |  |  |
| Leakage? | preprocessing / train-test contamination | source, time, event, near-duplicate |  |  |
| Label issues? | sentiment conflict | category ambiguity / multi-topic |  |  |
| Language-specific issue? | HTML entities/tags | Unicode/encoding tiếng Việt |  |  |
| Metadata risk? | ít metadata | source/url/publish_date |  |  |

## Kết luận ngắn

Viết 5–7 câu chỉ ra:

- kiểm tra nào chuyển nguyên từ IMDB sang UVN-1;
- kiểm tra nào phải thay đổi cách diễn giải;
- vấn đề mới nào nổi bật khi làm việc với tin tức tiếng Việt.
