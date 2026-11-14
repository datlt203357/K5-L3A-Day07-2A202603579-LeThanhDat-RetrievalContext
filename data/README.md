# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Tái tạo corpus Day 6 dùng cho Day 7

Corpus benchmark là Sổ tay sinh viên 2026 của Đại học Bách khoa Hà Nội. Nguồn công bố được ghi trong manifest Day 6: <https://ctsv.hust.edu.vn/tan-sinh-vien/nhung-dieu-can-biet>. Trước khi tải hoặc chia sẻ, người thực hiện phải dùng nguồn truy cập hợp pháp và tuân thủ điều kiện/quyền của chủ sở hữu. Manifest Day 6 hiện ghi chưa có giấy phép mở và chưa có quyền tái phân phối PDF, toàn văn chunks hoặc trace chứa nhiều nội dung nguồn.

Nếu người chấm đã có bản PDF được phép sử dụng và repo Day 6:

1. Đặt PDF tại `D:\Workspace\AI20K\Lab Day 6\So tay sinh vien_2026.pdf` hoặc cập nhật `data/manifest.json` trong Day 6 cho đúng vị trí. Giữ các trang được chọn, quy tắc chuẩn hóa, sửa lỗi trích xuất và cách chunk như manifest/ingestion Day 6.
2. Đảm bảo PostgreSQL + pgvector Day 6 đang chạy, `DATABASE_URL` và `EMBEDDING_MODEL` được cấu hình trong Day 6 `starter/.env`, rồi chạy ingestion từ thư mục Day 6 `starter`:

   ```powershell
   & 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' -m src.run
   ```

3. Từ thư mục gốc Day 7, xuất text/ID/metadata của strategy B (vector không được xuất vì Day 7 tái dùng vector store):

   ```powershell
   & 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\export_day6_chunks.py --strategy B --output data\day6_chunks_B.json
   ```

Nếu Day 6 nằm ở vị trí khác, thêm `--day6-root <đường-dẫn-Day-6>`. Script đọc `DATABASE_URL` từ Day 6 `.env` và xuất những cột chunk cần cho retrieval; không xuất embedding. Chỉ cung cấp file JSON xuất ra cho người chấm nếu được chủ sở hữu dữ liệu cho phép. Nếu chưa có quyền chia sẻ, giữ dữ liệu local và thu xếp quyền truy cập/cấp bản corpus hợp pháp với người chấm; tài liệu truy cập công khai không tự động đồng nghĩa có quyền tái phân phối.
