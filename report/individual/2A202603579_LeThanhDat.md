# Báo cáo Day 7 — 2A202603579 - Lê Thành Đạt

## Dữ liệu và benchmark

- **Nguồn:** Sổ tay sinh viên 2026, Đại học Bách khoa Hà Nội; nguồn chính thức: `https://ctsv.hust.edu.vn/tan-sinh-vien/nhung-dieu-can-biet`. PDF có 55 trang vật lý; dùng 52 trang nội dung (trang 2–3 và 5–54). Strategy B có 444 chunks, giữ `chunk_id`, `document_id`, `source`, trang PDF vật lý và metadata `section`.
- **Quyền sử dụng:** Theo manifest Day 6, chưa xác nhận giấy phép mở hoặc quyền tái phân phối. Corpus được dùng local cho lab theo điều kiện đã ghi trong manifest; không đưa PDF, toàn văn chunks hoặc trace chứa nhiều nội dung nguồn lên repo công khai khi chưa có quyền.
- **Vector store:** PostgreSQL 16.16 + pgvector 0.8.7, local Docker, bảng `day6_chunks`, cosine distance. Day 7 kết nối qua `PgVectorStore` trong repo Day 6, dùng `DATABASE_URL` tại `D:\Workspace\AI20K\Lab Day 6\starter\.env`.
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, Sentence Transformers 6.3.0, 384 chiều; document và query dùng cùng model, vector chuẩn hóa. Embedder là `MyEmbedder` của Day 6. Model chạy local, chi phí API bằng 0. Phiên bản/revision model chưa được pin bằng hash.
- **Agent model:** `gpt-4o-mini` qua OpenAI-compatible; dùng `OPENAI_API_KEY` trong `.env` . Tool calling qua Chat Completions.
- **Benchmark:** 24 câu: 15 dev có đáp án + 2 dev thiếu evidence; 5 test có đáp án + 2 test thiếu evidence. Câu hỏi có đáp án lưu tại `data/benchmark-day7/questions.json`; câu thiếu evidence tại `data/benchmark-day7/unanswerable.json`; danh sách case tại `data/benchmark-day7/cases.json`. Bộ test Day 7 dùng câu mới, không lặp câu test Day 6. Test Day 7 chưa được dùng để chọn cấu hình.
- **Nhãn evidence:** Với câu có đáp án, evidence quote được đối chiếu với trang PDF vật lý và chunk đúng trang/nguồn. Câu thiếu evidence có lý do vắng mặt được ghi trong `unanswerable.json`. Gold chỉ dùng ở bước chấm sau retrieval, không truyền vào query/retriever.

## Retrieval và thử nghiệm

Các chỉ số retrieval dev dùng 15 câu answerable; bảng test-only dùng cùng định nghĩa trên 5 câu answerable. `Hit@1` là số câu có evidence hợp lệ ở hạng 1; `Evidence Hit@3` là số câu có evidence trong top 3; `Context Evidence Rate` là số câu có evidence trong context sau đóng gói chia số câu answerable của split. Context budget tính theo ký tự. Thời gian là median ms mỗi query; retrieval chạy local, API cost = 0.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Baseline | TF-IDF, top-3, không filter, budget 800 ký tự | Lexical baseline | dev | 9/15 (60,0%) | 11/15 (73,3%) | 10/15 (66,7%) | 2,41 ms/query; $0 API |
| Phương pháp 2 | Dense pgvector Day 6 strategy B, candidate-k 10, đánh giá top-3, budget 800 | Dùng embedding tương thích index Day 6 | dev | 6/15 (40,0%) | 11/15 (73,3%) | 8/15 (53,3%) | 61,09 ms/query*; $0 API |
| Phương pháp 3 | Hybrid dense + TF-IDF, RRF k=60, candidate-k 10 mỗi retriever, top-3, budget 800 | Kết hợp ranking dense và lexical | dev | 9/15 (60,0%) | 13/15 (86,7%) | 11/15 (73,3%) | 63,68 ms/query*; $0 API |
| Cải thiện 1 | Hybrid RRF k=60, candidate-k 20, top-3, budget 800 | Candidate rộng hơn có thể đưa evidence vào top-3 | dev | 9/15 (60,0%) | 12/15 (80,0%) | 11/15 (73,3%) | 87,60 ms/query*; $0 API |
| Cải thiện 2 | Hybrid RRF k=60, candidate-k 10, top-3, budget 1200 | Context lớn hơn có thể giữ thêm evidence | dev | 9/15 (60,0%) | 13/15 (86,7%) | 11/15 (73,3%) | 78,66 ms/query*; $0 API |
| Cấu hình retrieval được chọn | Hybrid RRF k=60, candidate-k 10, top-3, budget 800 | Tốt nhất trong các cấu hình retrieval-only dev đã đo theo Hit@3 và Context Evidence Rate; dùng làm retriever cho agent | dev | 9/15 (60,0%) | 13/15 (86,7%) | 11/15 (73,3%) | 63,68 ms/query*; $0 API |
| Retrieval-only test — dense | Day 6 strategy B, candidate-k 10, top-3, budget 800 | Đo trên test sau khi khóa cấu hình; chỉ báo cáo, không dùng chọn cấu hình | test | 2/5 (40,0%) | 2/5 (40,0%) | 2/5 (40,0%) | 55,79 ms/query; $0 API |
| Retrieval-only test — TF-IDF baseline | TF-IDF, top-3, budget 800 | Baseline trên cùng test đã khóa | test | 4/5 (80,0%) | 4/5 (80,0%) | 4/5 (80,0%) | 2,92 ms/query; $0 API |
| Retrieval-only test — hybrid đã chọn | Hybrid RRF k=60, candidate-k 10, top-3, budget 800 | Cùng corpus, test và cutoff; không dùng test để đổi cấu hình | test | 2/5 (40,0%) | 3/5 (60,0%) | 2/5 (40,0%) | 59,35 ms/query; $0 API |

\* Median dense bao gồm query embedding và pgvector search; hybrid bao gồm query embedding, cả hai retrieval và RRF; TF-IDF chỉ tính lexical retrieval. Đây là số đo local của một lượt chạy, không quy đổi chi phí phần cứng. Trace/metadata dev: `starter/results/d7-retrieval-dev-02/`; hai cấu hình dev bổ sung: `starter/results/d7-hybrid-k20-dev/`, `starter/results/d7-hybrid-budget1200-dev/`; retrieval-only test sau khi khóa: `starter/results/d7-retrieval-test-locked/`. Test cho thấy TF-IDF retrieval-only cao hơn hybrid trên bộ câu nhỏ này; kết quả được giữ như đánh giá độc lập, không dùng để đổi cấu hình đã khóa.

**Lỗi dev đã xem và xử lý:**

1. `d7-dev-04` hỏi địa chỉ Tổ Tư vấn và Hỗ trợ Sinh viên. Lượt đầu lấy được các đoạn giới thiệu trên trang 21 nhưng bỏ sót dòng địa chỉ ở chunk riêng. Đã bổ sung lấy dòng địa chỉ cùng trang dựa trên trang do retrieval suy ra; lượt agent cuối trả lời đúng và cite `hust_handbook_2026:p21:paragraph:0007`.
2. `d7-dev-13` hỏi địa chỉ Văn phòng Quản lý ký túc xá. Lượt đầu ưu tiên nội dung KTX trang 46 và bỏ sót khối danh bạ trang 29. Query tìm địa chỉ ở lượt sau đưa chunk danh bạ vào context; lượt agent cuối trả lời đúng và cite `hust_handbook_2026:p29:paragraph:0003:token:000`.
3. Trong lượt agent đầu, `d7-dev-03`, `11`, `12`, `15` bị từ chối dù trace có evidence phù hợp hoặc chunk continuation. Đã thêm bước rà câu từ chối/citation, giữ chunk nối tiếp và yêu cầu cite các chunk liên quan. Lượt dev cuối trả lời đúng cả bốn câu; kết quả đối chiếu ở `starter/results/agent-dev-gpt4o-mini-review-final/manual_review.json`.

Hybrid candidate-k 10 được chọn cho retrieval-only trên dev vì tăng Evidence Hit@3 từ 11/15 của TF-IDF lên 13/15 và Context Evidence Rate từ 10/15 lên 11/15. Candidate-k 20 giảm Hit@3 xuống 12/15; budget 1200 không tăng Context Evidence Rate so với budget 800. Agent dùng cùng retriever nhưng budget 1600 để giữ evidence đủ cho trả lời. Test retrieval-only cho kết quả TF-IDF 4/5 Context Evidence Rate, hybrid 2/5 tại budget 800. Đây là khác biệt trên test đã khóa, được báo cáo nguyên trạng và không dùng để đổi cấu hình.

**Lệnh tái lập dev:**

```powershell
python starter\baseline.py --chunks data\day6_chunks_B.json --questions data\benchmark-day7\questions.json --split dev --output starter\results\d7-b-check
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\day6_retrieval.py --split dev --strategy B --candidate-k 10 --budget 800 --output starter\results\d7-retrieval-dev-02
```

**Lệnh đánh giá test sau khi khóa cấu hình:**

```powershell
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\run_agent_benchmark.py --split test --retriever tfidf --max-searches 2 --top-k 3 --candidate-k 10 --budget 1600 --output starter\results\agent-test-gpt4o-mini-tfidf
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\run_agent_benchmark.py --split test --retriever hybrid_rrf --max-searches 2 --top-k 3 --candidate-k 10 --budget 1600 --output starter\results\agent-test-gpt4o-mini-final
& 'D:\Workspace\AI20K\Lab Day 6\starter\.venv\Scripts\python.exe' starter\day6_retrieval.py --split test --strategy B --candidate-k 10 --budget 800 --output starter\results\d7-retrieval-test-locked
```

## Agent và đánh giá

- Agent đã triển khai với tool `search_evidence`, hybrid RRF (dense Day 6 + TF-IDF), tối đa 2 lượt tìm, top-3 kết quả mỗi lượt. Prompt yêu cầu chỉ trả lời theo evidence, trích dẫn chunk ID và từ chối khi không có bằng chứng đủ dùng.
- Lượt dev được rà soát chạy trên 17 case bằng `gpt-4o-mini`: trả lời có citation 15/15 câu có đáp án; cả 15 câu đạt grounded answer 2/2, trung bình 2,0. Có 21 tool calls; median 4.588 ms/query; dùng 27.049 prompt tokens và 1.739 completion tokens (28.788 tổng). Chấm từng citation: 20/24 citation hỗ trợ claim đi kèm; 15/15 câu trả lời có ít nhất một citation. Hai câu thiếu evidence được từ chối đúng (2/2). Bảng nhãn từng citation trong `starter/results/agent-dev-gpt4o-mini-review-final/reviews.json` và `summary.json`. Chi phí ước tính khoảng **$0,00510–$0,00612**; số thực tế phụ thuộc nhóm giá/key và hóa đơn của tài khoản. Chi tiết phép tính ở `cost_estimate.json`.
- Đã xử lý các lỗi trước đó: thêm truy xuất dòng địa chỉ/lịch trên trang đã được retriever nhận diện; giữ chunk continuation khi evidence bị tách; cho model rà lại câu từ chối hoặc citation không hợp lệ; giới hạn câu trả lời vào điều được hỏi và tự gắn citation cho các chunk continuation cùng đoạn.
- Trace và metadata dev: `starter/results/agent-dev-gpt4o-mini-review-final/`; đối chiếu answer/citation với gold: `starter/results/agent-dev-gpt4o-mini-review-final/manual_review.json`.
- Cấu hình agent được khóa trước test: hybrid RRF, candidate-k 10, top-3, tối đa 2 lượt search, context budget 1600. Chọn theo kết quả retrieval dev, độ phủ context, câu trả lời grounded và chi phí/thời gian; test không được dùng để chọn cấu hình.
- Sau khi khóa, chạy cả TF-IDF baseline agent và hybrid agent trên cùng 7 test case, không đưa gold vào query/model và không điều chỉnh sau khi xem test. TF-IDF baseline trả lời 4/5 câu có đáp án (1 false refusal); hybrid trả lời 3/5 (2 false refusal). Cả hai từ chối đúng 2/2 câu thiếu evidence.
- Context được chấm 0/1/2 cho từng câu answerable: baseline `[2,2,2,2,1]`, trung bình 1,8/2; hybrid `[0,2,2,2,1]`, trung bình 1,4/2. Baseline có 3/4 citation hỗ trợ claim; hybrid có 4/5. Answer citation coverage lần lượt 4/4 và 3/3 câu answered. Baseline có median 4.865 ms/query, 9 tool calls, 12.300 prompt + 756 completion tokens (13.056 tổng); hybrid có median 5.563 ms/query, 11 tool calls, 12.717 prompt + 684 completion tokens (13.401 tổng). Theo cùng khoảng giá công khai, chi phí ước tính baseline **$0,00230–$0,00276**, hybrid **$0,00232–$0,00278**; hóa đơn thực tế phụ thuộc nhóm giá/key. Phép tính lưu cùng artifact của mỗi run.
- Trace, reviews và summary baseline: `starter/results/agent-test-gpt4o-mini-tfidf/`; hybrid: `starter/results/agent-test-gpt4o-mini-final/`. Mỗi thư mục có `traces.json`, `reviews.json` (nhãn từng citation), `manual_review.json` (context grade/lý do), `summary.json` và `run_metadata.json`.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| GPT-4o mini, hybrid RRF, dev, max-searches 2, top-3, budget 1600 | Search evidence | 21 / 17 case | 2,0 (15/15 đạt 2) | 20/24 citation | 15/15 | 2/2 unanswerable | 0/15 answerable |
| GPT-4o mini, TF-IDF baseline, test, max-searches 2, top-3, budget 1600 | Search evidence | 9 / 7 case | 1,4 (3 đạt 2; 1 đạt 1; 1 đạt 0) | 3/4 citation | 4/4 | 2/2 unanswerable | 1/5 answerable |
| GPT-4o mini, hybrid RRF đã chọn, test, max-searches 2, top-3, budget 1600 | Search evidence | 11 / 7 case | 1,2 (3/5 đạt 2; 2/5 đạt 0) | 4/5 citation | 3/3 | 2/2 unanswerable | 2/5 answerable |

| Query ID | Context grade TF-IDF | Context grade hybrid | Ghi chú |
|---|---:|---:|---|
| d7-test-01 | 2 | 0 | TF-IDF lấy đúng evidence; hybrid bỏ sót câu giải thích trong top-3/context. |
| d7-test-02 | 2 | 2 | Cả hai context có địa điểm Thư viện Tạ Quang Bửu. |
| d7-test-03 | 2 | 2 | Cả hai context có đoạn diễn đạt đơn giản/dễ hiểu của bạn cùng lứa. |
| d7-test-04 | 2 | 2 | Cả hai context có hai tỷ lệ 15% và 85%. |
| d7-test-05 | 1 | 1 | Có đoạn kỹ năng lập kế hoạch dự án liên quan nhưng thiếu đủ ba kỹ năng yêu cầu. |
| **Trung bình / 5** | **1,8** | **1,4** | Context được chấm 0 = thiếu evidence, 1 = evidence liên quan nhưng thiếu ý, 2 = đủ evidence. |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| 17 case dev | Xem trace và `manual_review.json` trong `starter/results/agent-dev-gpt4o-mini-review-final/` | Đã đối chiếu từng câu | 15 answerable có citation đúng; 2 unanswerable từ chối đúng | Cả 15 answerable có gold evidence trong chunk được cite |
| 7 case test | Xem trace/reviews trong thư mục baseline TF-IDF và hybrid | Đã đối chiếu từng câu và từng citation | Baseline grounded mean 1,4; hybrid 1,2; false refusal lần lượt 1/5 và 2/5 | 2/2 unanswerable được từ chối đúng ở cả hai cấu hình |

### Ví dụ câu trả lời, từ chối và lỗi

1. **Câu có bằng chứng — `d7-test-04`.** Câu hỏi: “Theo con số được sổ tay dẫn lại, tỷ lệ thành công gắn với kiến thức chuyên môn và kỹ năng mềm lần lượt là bao nhiêu?” Agent trả lời: “Tỷ lệ thành công gắn với kiến thức chuyên môn là 15%, trong khi tỷ lệ gắn với kỹ năng mềm là 85%” `[hust_handbook_2026:p44:paragraph:0002]`. Chunk trang 44 nêu trực tiếp cả hai tỷ lệ.
2. **Câu thiếu bằng chứng — `d7-test-u01`.** Câu hỏi: “Mức học phí chính thức trên mỗi tín chỉ của Trường Cơ khí trong năm học 2026–2027 là bao nhiêu?” Tài liệu không nêu biểu phí theo tín chỉ cho Trường Cơ khí hoặc mức học phí năm học đó; agent trả lời “Không đủ bằng chứng để trả lời.” Đây là từ chối đúng.
3. **Lỗi retrieval — `d7-test-01`.** Câu hỏi: “Vì sao cần nắm chắc kiến thức cơ bản trước khi làm câu hỏi khó?” Hybrid đã chọn không đưa câu evidence vào context nên agent từ chối nhầm. TF-IDF baseline lấy được đúng chunk và trả lời đúng ý chính, nhưng thêm một khẳng định không có trong evidence; vì vậy baseline được chấm grounded 1/2 và citation của câu đó không đạt.

## Thử thách nâng cao

Chưa thực hiện.
