# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Repo hiện có runner agent thật dùng GPT-4o mini qua endpoint OpenAI-compatible, dense retrieval trên vector store Day 6, TF-IDF và hybrid RRF. `agentic.py` vẫn là khung ví dụ; các lệnh benchmark bên dưới dùng `run_agent_benchmark.py` và adapter trong repo. Mock trong unit test chỉ kiểm tra logic, không phải kết quả benchmark.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

### Dùng GPT-4o mini

Đã có adapter `openai_controller.py` và runner `run_agent_benchmark.py`. Cài OpenAI Python SDK vào virtual environment Day 6 (runner cần chung môi trường với pgvector/Sentence Transformers):

```powershell
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' -m pip install openai
```

Tạo `.env` từ mẫu ở **thư mục gốc Day 7**. File `.env` đã được Git ignore. Điền API key ShopAIKey của bạn tại máy, không commit hoặc gửi key trong chat:

```powershell
Copy-Item .env.example .env
notepad .env
```

Giữ các giá trị `OPENAI_BASE_URL` và `AGENT_MODEL` trong mẫu; chỉ thay `YOUR_SHOPAIKEY_API_KEY` bằng key của bạn.

Trước khi chạy benchmark, có thể gửi một yêu cầu ngắn để kiểm tra endpoint/key/model. Yêu cầu này dùng API và có thể phát sinh phí:

```powershell
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' -c "from dotenv import load_dotenv; import os; from openai import OpenAI; load_dotenv('.env'); c=OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url=os.getenv('OPENAI_BASE_URL')); r=c.chat.completions.create(model=os.getenv('AGENT_MODEL'),messages=[{'role':'user','content':'Reply with OK only.'}],max_tokens=8); print(r.choices[0].message.content)"
```

Sau khi smoke check trả về `OK`, chạy agent trên **dev** trước:

```powershell
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\run_agent_benchmark.py --split dev --retriever hybrid_rrf --max-searches 2 --top-k 3 --candidate-k 10 --budget 1600 --output starter\results\agent-dev-gpt4o-mini
```

Runner chỉ đưa `id` và câu hỏi vào agent; gold/evidence không được gửi lên API. Nó ghi `traces.json` và `run_metadata.json`, gồm tool calls, context, câu trả lời, citations, lỗi và token usage. Chi phí thực tế tùy nhóm giá của key; đối chiếu token usage với bảng giá tài khoản. Báo cáo có thể kèm ước tính theo [bảng giá model ShopAIKey](https://shopaikey.com/models), nhưng ước tính công khai không thay thế hóa đơn. GPT-4o mini hỗ trợ function calling; key ShopAIKey cũng cần được cấp quyền cho đúng model ID. Nếu nhận lỗi model không có quyền, kiểm tra nhóm model của key và tên ID trong danh mục ShopAIKey.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 2a. So sánh Day 6 dense, TF-IDF và hybrid RRF

`day6_retrieval.py` nối trực tiếp `MyEmbedder` và `PgVectorStore` từ Day 6. Chạy từ repo Day 7 bằng Python trong virtual environment Day 6:

```powershell
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\day6_retrieval.py --split dev --strategy B --candidate-k 10 --budget 800 --output starter\results\d7-retrieval-dev
```

Script dùng model và `DATABASE_URL` trong `D:\Workspace\AI20K\Lab Day 6\starter\.env`, kiểm tra vector dimension 384, rồi ghi `summary.json`, `traces.json` và `run_metadata.json`. Ba phương pháp được so sánh là dense pgvector, TF-IDF và hybrid Reciprocal Rank Fusion (RRF, k=60). Gold chỉ được dùng sau retrieval để chấm evidence.

Tham số thử trên dev, mỗi lần dùng thư mục output mới:

```powershell
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\day6_retrieval.py --split dev --strategy B --candidate-k 20 --budget 800 --output starter\results\d7-hybrid-k20-dev
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\day6_retrieval.py --split dev --strategy B --candidate-k 10 --budget 1200 --output starter\results\d7-hybrid-budget1200-dev
```

Chỉ chạy `--split test` sau khi đã chốt cấu hình dựa trên dev. Nếu Day 6 ở vị trí khác, đặt biến môi trường `DAY6_ROOT` tới thư mục repo Day 6 trước khi gọi script.

Để tái tạo so sánh test đã khóa, chạy lexical-only agent baseline và hybrid agent với cùng model, top-k, context budget và benchmark; đồng thời chạy retrieval-only so sánh ba phương pháp. Các lệnh test chỉ đánh giá, không dùng kết quả để đổi cấu hình:

```powershell
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\run_agent_benchmark.py --split test --retriever tfidf --max-searches 2 --top-k 3 --candidate-k 10 --budget 1600 --output starter\results\agent-test-gpt4o-mini-tfidf
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\run_agent_benchmark.py --split test --retriever hybrid_rrf --max-searches 2 --top-k 3 --candidate-k 10 --budget 1600 --output starter\results\agent-test-gpt4o-mini-hybrid
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\day6_retrieval.py --split test --strategy B --candidate-k 10 --budget 800 --output starter\results\d7-retrieval-test-locked
```

Sau khi đọc context, câu trả lời và từng citation, ghi một nhãn `citation_support` cho **mỗi** citation, `answer_grade` (0–2) cho câu có đáp án, `abstention_correct` cho câu thiếu evidence, và `context_grade` (0–2) cho 5 câu test có đáp án. Dùng `agent_metrics.py` để tổng hợp các nhãn; không để script tự đoán độ đúng.

Nếu cần tái tạo `data/day6_chunks_B.json` từ vector store Day 6, xem [hướng dẫn corpus](../data/README.md#tái-tạo-corpus-day-6-dùng-cho-day-7) và chạy `starter/export_day6_chunks.py`. File xuất chứa toàn văn nguồn và chỉ được chia sẻ khi có quyền.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
