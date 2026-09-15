# Project Memory: AI Phân loại Nhạc Bản quyền (Music Rights AI)

Tài liệu này đóng vai trò là **bộ nhớ ngữ cảnh liên tục** (Persistent Memory) cho Claude Code và các AI assistant khi làm việc trong repository qua các phiên làm việc khác nhau. Cập nhật tài liệu này khi có quyết định kiến trúc mới hoặc thay đổi trạng thái milestone.

---

## 1. Thông tin Dự án Cốt lõi
- **Tên dự án**: Hệ thống AI Phân loại Nhạc Bản quyền (AI-based Music Identification & Copyright Usage Risk Assessment System).
- **Mô hình cốt lõi**: Cascade Pipeline nối tiếp:
  `Input → Preprocessing → Chromaprint (AI-1) → [nếu < τFP] MERT-v1-95M (AI-2) → Rights Lookup → Rule Engine → Risk & Recommendation`.
- **Tech Stack đã chốt**:
  - Backend: FastAPI (Python 3.10+)
  - Cơ sở dữ liệu: PostgreSQL (metadata, rights, history)
  - Vector search: FAISS (nghiên cứu) → Qdrant (triển khai sản phẩm)
  - Audio tools: FFmpeg, Chromaprint (`fpcalc`)
  - AI Models: `MERT-v1-95M` (Mean / Mean+Std pooling), CQT/Chroma baseline cho Cover
  - Container hóa: Docker / Docker Compose

---

## 2. Trạng thái Hiện tại & Tiến độ (Current Status)

- **Milestone đang thực hiện**: **`M6 — Full Application Ready`** / chuẩn bị nghiệm thu **`M7 — Final Evaluation`**.
- **Hạng mục đã hoàn thành**:
  - [x] Thiết lập hệ thống Modular Rules chuẩn trong `.claude/rules/` (7 files).
  - [x] Xây dựng CSDL tham chiếu — **đã mở rộng lên 138.164 compositions/recordings/rights** (từ mốc 4.922 ban đầu; xem cảnh báo quy mô artifact bên dưới).
  - [x] Tầng 1 (Chromaprint): Codec thuần Python, vectorised bit-error matching, τFP = 0.15 (`EXP-01`).
  - [x] Tầng 2 (MERT Deep Retrieval): MERT-v1-95M + FAISS FlatIP, τMERT = 0.97 (`EXP-02`, `EXP-03`, `EXP-06`).
  - [x] Tầng 3 (Cover Service): CQT + Optimal Transposition Index, τCover = 0.97 (`EXP-07`) — **đã nối chính thức vào `cascade_service.py`** (nhánh `STAGE_3_COVER`, bật qua `config.COVER_ENABLED`).
  - [x] Rule Engine: Cây quyết định tuần tự 5 nhóm bản quyền theo `configs/rules_v1.yaml` (65 unit tests PASS).
  - [x] License Classifier & Production Features: `EXP-09`, trích xuất RMS/Crest factor/Dynamic range.
  - [x] REST API FastAPI & Giao diện tĩnh (frontend/index.html+app.js+styles.css, mount trực tiếp ở `/` từ `backend/main.py`, không cần build).
  - [x] Chạy hoàn tất 9/9 thí nghiệm (`EXP-01` đến `EXP-09`), lưu kết quả trong `experiments/results/` — **vượt 8 thí nghiệm gốc trong `06-experiments-evaluation.md`** (chưa đồng bộ tài liệu, xem known-gap bên dưới).
  - [x] Bộ test tự động: 174 tests PASS (đã xử lý tương thích torchvision cho MERT).
  - [x] **`git init` + commit đầu tiên** (`f4cc43f`, 120 file, `.git` ~75MB) — đã audit an toàn: không secret (`.env` biến thể), không file >50MB, `.gitignore` loại đúng dữ liệu sinh-lại-được (`data/preprocess/`, `data/processed/_archive_*/`, `data/test_queries/`, config `*.local.*`, audio test mẫu).
  - [x] Response schema API (`backend/schemas/analysis.py`) — mọi field đã có `Field(description=...)`, kể cả `evidence` (mô tả cấu trúc dict động: `identification`/`candidates`/`rule_engine`/`composition`/`rights_record`).

- **⚠️ Gap mới phát sinh sau khi mở rộng corpus lên 138k — artifact dẫn xuất CHƯA đuổi kịp**:
  | Bảng | Số dòng |
  |---|---|
  | compositions/metadata/rights_master.csv | **138.164** |
  | fingerprints_master.csv | 104.000 |
  | embeddings_master.csv | 8.000 |
  | test_queries_master.csv | 2.150 |
  | cover_descriptors.npy | ~100 reference window |

  Threshold hiện tại (τFP/τMERT/τCover) được hiệu chỉnh trên quy mô nhỏ hơn nhiều — `config.py` tự cảnh báo: ngưỡng phụ thuộc quy mô reference, mở rộng corpus mà không hiệu chỉnh lại là sai.

- **Hạng mục tiếp theo (Next Actions)**:
  - [ ] Chạy `scripts/build_embeddings.py --all` rồi `scripts/rebuild_faiss_index.py` để embedding/FAISS đuổi kịp corpus 138k (hiện mới 8k).
  - [ ] Sinh lại `test_queries_master.csv` qua `scripts/augment_audio.py` cho tương xứng quy mô mới, rồi **chạy lại EXP-01/04/05/06/08** để threshold còn đúng.
  - [ ] Mở rộng `cover_descriptors.npy` (hiện ~100 window) để Stage 3 Cover phát huy tác dụng trên toàn corpus.
  - [ ] Kết nối PostgreSQL và chạy `python init_db.py` để hoàn tất dữ liệu runtime cho API (chưa xác nhận đã chạy ở quy mô 138k).
  - [ ] Đánh giá và tối ưu hóa UI/UX Frontend trên trình duyệt.
  - [ ] (chưa làm) 3 nguồn dataset ngoài FMA/Jamendo: YouTube Audio Library, Public Domain, Cover dataset thật (SecondHandSongs subset).
  - [ ] (known-gap, chưa đồng bộ) `.claude/rules/06-experiments-evaluation.md` vẫn ghi "8 thí nghiệm EXP-01..08" — chưa thêm EXP-09.

---

## 3. Ràng buộc & Quyết định Kỹ thuật Bất biến (Key Invariants)

1. **Composition ID ≠ Recording ID**: Một composition có nhiều recording; tuyệt đối không suy luận composition PD ⇒ recording PD.
2. **Cây quyết định tuần tự**: `Audio Library → Creator Music → Content ID → Creative Commons → Public Domain → Fallback`.
3. **Không dùng ML cho đánh giá pháp lý**: Rule Engine là logic thuần `if/else`, cấu hình qua YAML/JSON. Cấm dùng CLAP làm "copyright classifier".
4. **Xử lý UNKNOWN minh bạch**: Không ép UNKNOWN thành LOW/HIGH. Trường hợp nghi ngờ luôn yêu cầu con người thẩm định (`HUMAN_REVIEW_REQUIRED`).
5. **Data Split**: Bắt buộc split tập train/val/test theo `recording_id` (hoặc `composition_id` với cover), tuyệt đối không để chung bài ở cả hai tập.
6. **Ưu tiên Core MVP**: Không làm Cover nâng cao, RAG, LLM hay fine-tuning trước khi core cascade pipeline chạy ổn định end-to-end.

---

## 4. Tham chiếu nhanh Hệ thống Rules (`.claude/rules/`)

- [01-core-invariants.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/01-core-invariants.md): 9 nguyên tắc bất biến & phạm vi ngoài MVP.
- [02-pipeline-architecture.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/02-pipeline-architecture.md): Luồng cascade tiền xử lý và model AI.
- [03-rule-engine-copyright.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/03-rule-engine-copyright.md): Cây quyết định 5 nhóm bản quyền & fallback.
- [04-database-schemas.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/04-database-schemas.md): Đặc tả 7 bảng cơ sở dữ liệu PostgreSQL.
- [05-backend-api.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/05-backend-api.md): Cấu trúc service, API endpoints, mã lỗi & logging.
- [06-experiments-evaluation.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/06-experiments-evaluation.md): 8 thí nghiệm (EXP-01..08), dataset & ngưỡng nghiệm thu.
- [07-ui-and-roadmap.md](file:///d:/PycharmProjects/AMR_advanced/.claude/rules/07-ui-and-roadmap.md): 4 màn hình giao diện & lộ trình 12 tuần.

---

## 5. Danh mục Custom Subagents (`.claude/agents/`)

- [researcher.md](file:///d:/PycharmProjects/AMR_advanced/.claude/agents/researcher.md) (`model: sonnet`): Chuyên gia nghiên cứu MIR, MERT embedding, Chromaprint, dataset & phương pháp thực nghiệm.
- [reviewer.md](file:///d:/PycharmProjects/AMR_advanced/.claude/agents/reviewer.md) (`model: opus`): Kiểm soát viên kiến trúc, thẩm định chất lượng code, API, schema database, và 9 nguyên tắc bất biến.
- [qa-tester.md](file:///d:/PycharmProjects/AMR_advanced/.claude/agents/qa-tester.md) (`model: sonnet`): Kỹ sư kiểm thử tự động hóa & QA, phụ trách test suite 50–100 cases Rule Engine, API integration tests và kiểm thử sức bền âm thanh (robustness).

---

## 6. Danh mục Skills Dự án (`.claude/skills/`)

- [music-rights-pipeline](file:///d:/PycharmProjects/AMR_advanced/.claude/skills/music-rights-pipeline/SKILL.md): Playbook vận hành, tích hợp và debug luồng Cascade Audio Processing (FFmpeg → Chromaprint → MERT → Retrieval → Rights).
- [rule-engine-verifier](file:///d:/PycharmProjects/AMR_advanced/.claude/skills/rule-engine-verifier/SKILL.md): Playbook thẩm định, mở rộng và kiểm thử Rule Engine 5 nhóm bản quyền (unit test suite 50–100 cases).
- [experiment-benchmark](file:///d:/PycharmProjects/AMR_advanced/.claude/skills/experiment-benchmark/SKILL.md): Playbook thực thi, đo lường và đánh giá 8 bài thí nghiệm chuẩn (EXP-01..08) cùng bảng nghiệm thu nội bộ.

---

## 7. Nhật ký Phiên làm việc (Session Log)

- **2026-09-15**:
  - Hoàn tất phân rã `CLAUDE.md` thành 7 file quy tắc modular trong `.claude/rules/`.
  - Khởi tạo file bộ nhớ `memory.md` để lưu trữ ngữ cảnh liên tục cho dự án.
  - Cập nhật file `settings.json` bổ sung schema và quyền thực thi toàn diện.
  - Tạo file `CLAUDE.local.md` cấu hình cục bộ cho môi trường phát triển Windows.
  - Tạo 3 custom subagents trong `.claude/agents/`: `researcher.md` (Sonnet), `reviewer.md` (Opus) và `qa-tester.md` (Sonnet).
  - Tạo 3 skills chuyên biệt trong `.claude/skills/`: `music-rights-pipeline`, `rule-engine-verifier`, `experiment-benchmark`.
  - Thiết lập hệ thống Hooks tự động trong `.claude/hooks/format_and_lint.py` và cấu hình `PostToolUse` trong `settings.json`.
  - Tạo file `settings.local.json` cấu hình biến môi trường cục bộ và quyền thực thi riêng cho máy Windows.
  - Kích hoạt cơ chế Tự động đồng bộ & Cập nhật Rules, Skills và Memory (Self-Updating System) trong `01-core-invariants.md` và `format_and_lint.py`.

- **2026-09-15 (phiên đồng bộ nền tảng)**:
  - Phát hiện `memory.md` lỗi thời so với thực tế filesystem (vẫn ghi "M1 đang làm" trong khi đã có 9 experiment, 14 service, FAISS index, license classifier) — đồng bộ lại mục 2.
  - Rà soát `.gitignore` trước `git init`, phát hiện và chặn 3 gap nghiêm trọng: `data/preprocess/` (1.3GB dump FMA thô, gồm 2 file >100MB), `data/processed/_archive_*/` (390MB snapshot trùng lặp không bị chặn do pattern cũ có `/` neo path), `data/test_queries/` (2.1GB audio test sinh lại được). Thêm pattern `.env.*` bắt được 1 file `.env.bi-ghi-de-20260910` lọt qua rule `.env.bak*` cũ (nội dung an toàn, chỉ password mặc định dev). Loại thêm 2 file cấu hình riêng máy (`settings.local.json`, `CLAUDE.local.md`) và 2 file audio mẫu (`test.mp3`, `test_audio/`) theo quyết định của chủ dự án.
  - `git init` + rà soát staging trước commit — song song, một phiên khác đã tự hoàn tất bước này trước (commit `f4cc43f`, dùng đúng `.gitignore` vừa sửa cộng thêm 1 rule); đã audit lại xác nhận an toàn thay vì làm trùng.
  - Phát hiện quy mô corpus đã tăng từ 4.922 lên 138.164 bản ghi trong lúc thao tác (do một phiên khác chạy song song) — cập nhật lại mục 2 theo đúng số liệu mới nhất, thêm bảng cảnh báo artifact dẫn xuất (embedding/FAISS/test queries) chưa đuổi kịp quy mô.
  - Bổ sung `Field(description=...)` cho toàn bộ field trong `backend/schemas/analysis.py` (trước đó chỉ 2/20+ field có mô tả) để làm rõ tham số đầu ra API, đặc biệt `evidence` (dict động, mô tả bằng văn xuôi thay vì ép kiểu Pydantic lồng nhau để tránh lệch schema mỗi khi Rule Engine thêm `extra_evidence` mới).
