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
  - [x] Xây dựng CSDL tham chiếu — **158.117 compositions/recordings/rights**, trong đó 24.375 bài FMA medium có audio thật trên đĩa (từ mốc 4.922 ban đầu).
  - [x] Tầng 1 (Chromaprint): Codec thuần Python, vectorised bit-error matching, **τFP = 0.30** (`EXP-01` trên 24.375 fingerprint thật / 1.900 truy vấn).
  - [x] Tầng 2 (MERT Deep Retrieval): MERT-v1-95M + FAISS FlatIP, τMERT = 0.97 (`EXP-02`, `EXP-03`, `EXP-06`) — ⚠️ con số của corpus CŨ, chờ EXP-06 chạy lại.
  - [x] Tầng 3 (Cover Service): CQT + Optimal Transposition Index, **τCover = 0.90** (`EXP-07` đo đúng điều kiện server trên chỉ mục 24.375 bài) — **đã nối chính thức vào `cascade_service.py`** (nhánh `STAGE_3_COVER`, bật qua `config.COVER_ENABLED`).
  - [x] Rule Engine: Cây quyết định tuần tự 5 nhóm bản quyền theo `configs/rules_v1.yaml` (65 unit tests PASS).
  - [x] License Classifier & Production Features: `EXP-09`, trích xuất RMS/Crest factor/Dynamic range.
  - [x] REST API FastAPI & Giao diện tĩnh (frontend/index.html+app.js+styles.css, mount trực tiếp ở `/` từ `backend/main.py`, không cần build).
  - [x] Chạy hoàn tất 9/9 thí nghiệm (`EXP-01` đến `EXP-09`), lưu kết quả trong `experiments/results/` — **vượt 8 thí nghiệm gốc trong `06-experiments-evaluation.md`** (chưa đồng bộ tài liệu, xem known-gap bên dưới).
  - [x] Bộ test tự động: 174 tests PASS (đã xử lý tương thích torchvision cho MERT).
  - [x] **`git init` + commit đầu tiên** (`f4cc43f`, 120 file, `.git` ~75MB) — đã audit an toàn: không secret (`.env` biến thể), không file >50MB, `.gitignore` loại đúng dữ liệu sinh-lại-được (`data/preprocess/`, `data/processed/_archive_*/`, `data/test_queries/`, config `*.local.*`, audio test mẫu).
  - [x] Response schema API (`backend/schemas/analysis.py`) — mọi field đã có `Field(description=...)`, kể cả `evidence` (mô tả cấu trúc dict động: `identification`/`candidates`/`rule_engine`/`composition`/`rights_record`).

- **Quy mô artifact sau đợt nạp FMA medium (2026-09-16)**:
  | Bảng | Số dòng | Trạng thái |
  |---|---|---|
  | compositions/metadata/rights_master.csv | **158.117** | đã nạp vào PostgreSQL |
  | fingerprints_master.csv | 24.375 | fingerprint THẬT của fpcalc (149 bài trùng audio với bài khác) |
  | cover_descriptors.npy | 24.375 | mọi bản ghi có audio, 30 giây đầu |
  | data/test_queries/manifest.csv | 1.900 | 100 bài nguồn × 19 biến đổi |
  | embeddings_master.csv | **đang dựng** | ~5 s/bài trên CPU, phải chạy một mình |
  | test_queries_master.csv | cũ (2.150) | chờ `generate_test_queries.py` sau khi có embedding |

  133.742 bản ghi còn lại (nhạc Việt, Spotify, Jamendo…) là **metadata-only**: tra được qua `/api/v1/tracks/{id}` nhưng không nhận diện được qua audio.

- **Hạng mục tiếp theo (Next Actions)**:
  - [ ] `build_embeddings.py --all` (đang chạy) → `generate_test_queries.py` → `init_db.py` → **EXP-06** để hiệu chỉnh lại τMERT trên 24.375 bài.
  - [ ] `train_license_classifier.py --sweep` lại trên embedding mới (bản hiện tại là C=30 trên 4.000 bản ghi).
  - [ ] Độ trễ Tầng 1 tăng theo quy mô: 4,8 s/truy vấn với 24.375 fingerprint (0,185 ms/cặp × 24.375). Cần phương án trước khi demo.
  - [ ] EXP-03 chạy ở giao thức cũ (6.098 cửa sổ) nên EXP-07 không so được cột MERT — chạy lại nếu cần bảng so sánh.
  - [ ] (known-gap) EXP-04/EXP-05/EXP-08 chưa chạy lại ở quy mô 24.375.
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

- **2026-09-15 (phiên mở rộng dữ liệu & làm rõ kết quả — pipeline dài đang chạy)**:
  - `Chromaprint UNAVAILABLE` trong UI là do `.env` trỏ PostgreSQL cổng 5433 (Docker cũ, đã gỡ) trong khi máy chạy PostgreSQL 18 native cổng 5432 → `db=None`, không phải thiếu fpcalc (fpcalc có ở `~/bin`). Chủ dự án tự sửa `DATABASE_URL`. Nhánh UNAVAILABLE nay trả `reason_code` (DATABASE_UNAVAILABLE | FPCALC_MISSING), `threshold`, `fingerprint_score=null`.
  - Loại 100.000 fingerprint giả của `dataset_G_vietnam_100k_api` (nhạc Việt mới có metadata, CHƯA có audio; `audio_path` là placeholder `sp_mb_NNNNNN`). Thêm `chromaprint_codec.is_plausible_fingerprint`, check FAIL trong `check_data_integrity.py`, và chặn trong `process_all_datasets.py`.
  - Nhãn giấy phép 29.881 dòng Jamendo do `process_all_datasets.py` suy từ cờ `audiodownload_allowed` → `source` đổi sang tiền tố `SIMULATED`, `verified_at` rỗng; `train_license_classifier.py` loại nhãn SIMULATED/PREDICTED.
  - `ingest_corpus.py` từng ghi đè master và xoá mọi nguồn trừ AUDIO_LIBRARY/CREATOR_MUSIC/CONTENT_ID → sửa thành chỉ thay bản ghi cùng `source_dataset`.
  - Rule Engine: thêm `rights_gate` (min_rights_confidence 0.50) — quyền PREDICTED không còn ra LOW/CONDITIONAL/HIGH; evidence có `rights_confidence_breakdown`, `decision_confidence_formula`, `provisional_decision`.
  - Cover: nhiễu trắng đạt 0.985 > τCover → descriptor trừ trung bình từng khung (nhiễu còn 0.14, bài khác max 0.87); `cover_descriptors.npy` đã nâng cấp tại chỗ; τCover = 0.97 chờ EXP-07 hiệu chỉnh lại. EXP-07 thêm nhiễu làm mẫu âm.
  - EXP-06 tính theo khối (ma trận N×N ở 50k vector ≈ 10 GB, máy có 15,8 GB RAM).
  - License classifier: sweep C trên 4.000 bản ghi FMA — tổng lỗi thấp nhất ở C=30 (macro-F1 0.173 vs baseline 0.079; accuracy 0.293 < baseline 0.383).
  - Sửa lỗi có sẵn: `PipelineStage` thiếu `COVER_SEARCH` → mọi job chạy tới Tầng 3 làm `GET /jobs/{job_id}` trả 500 (thêm `tests/test_pipeline_stages.py`).
  - Kiểm thử trên server thật: nhiễu trắng → Tầng 1 0.086 < 0.15, MERT 0.935 < 0.97, Cover 0.166 → `UNKNOWN` qua `rights_gate`; bài FMA chưa có trong index bị model đoán `CC_BY_NC` → kết luận tạm HIGH được hạ về `UNKNOWN` kèm phép tính. Điểm Cover của nhạc thật trên thang mới: 0.41–0.57.
  - Đã tải xong FMA medium: 44 shard (22,4 GB), 24.801 file audio, 24.375 bài có giấy phép đọc được (`data/processed/corpus_fma_medium.csv`).
  - Độ trễ Tầng 1 theo quy mô: chi phí chính là tải + giải nén lại cả bảng fingerprint ở MỖI truy vấn (2,2 s + 2,9 s ở 4.000 dòng; ~40 s ở 28.000), không phải so khớp (0,185 ms/cặp). `fingerprint_service` nay cache bảng đã giải nén theo tiến trình (làm mới theo số dòng + min/max `fingerprint_id`). Đã thử so khớp theo khối tensor: đúng tuyệt đối nhưng CHẬM hơn (0,276 ms/cặp) nên bỏ.
  - EXP-01 tính điểm một lần cho mỗi truy vấn rồi dùng lại cho lượt held-out (trùng khớp bản cũ, nhanh gấp đôi).
  - Đang chạy: `ingest_corpus.py --sources fma_medium` → `init_db.py` → `build_embeddings.py --all` (CPU, ước ~1 ngày) → EXP-01/06/07 + sweep lại → cập nhật τ.

- **2026-09-16 (hiệu chỉnh lại ngưỡng trên corpus 24.375 bài có audio)**:
  - Nạp xong FMA medium: 158.117 bản ghi trong PostgreSQL, 24.375 fingerprint thật. `check_data_integrity.py`: 13 PASS / 2 WARN / 1 FAIL (FAIL ở FAISS id map là dự kiến tới khi dựng lại embedding).
  - **τFP 0.15 → 0.30** (EXP-01: P 0.9981, R 0.5558, FPR 0.0021). Bỏ quy tắc "lấy ngưỡng thấp nhất còn giữ FPR = 0" trong `apply_calibrated_thresholds.py`: quy tắc đó chọn 0.95 (recall 0.4679) để đổi lấy khác biệt mà dữ liệu không phân biệt được — 0 và 0.0011 chỉ cách nhau 2/1.900 truy vấn. Recall trần của tầng 1 là ~0.58 vì dịch cao độ và đổi tốc độ cho 0/100.
  - **τCover 0.97 → 0.90**. `build_cover_index.py` viết lại: lấy bài theo `audio_path` (bản cũ ghép theo TÊN FILE và ưu tiên chính file truy vấn trong `data/test_queries/`, nên chỉ mục chỉ có ~100 bài — đúng các bài đang dùng làm truy vấn, và mã bài đã chết sau khi nạp lại FMA). Có checkpoint, 24.375 bài trong 110 phút, không file nào lỗi.
  - EXP-07 thêm giao thức chấm **đúng điều kiện server** (`metrics.runtime_protocol`): 30 giây đầu của truy vấn, tìm trên toàn bộ chỉ mục, held-out loại cả bài trùng fingerprint. Ngưỡng cấp cửa sổ quá lạc quan: τ = 0.70 cho FMR 4,9% trên 100 bài nguồn nhưng 17,4% trên 24.375 bài. Ở τ = 0.90: P 0.9946, R 0.6837, FMR 0,4%, nhiễu 0% (nhiễu cao nhất 0.7278). Accuracy@1 0.8311, OTI đúng 91%, và nhóm dịch cao độ đạt 68% — đúng chỗ Chromaprint được 0%.
  - `pick_threshold` của EXP-07 siết FMR ≤ 0.005 trước, chỉ nới về trần 5% của §16 khi không có ngưỡng nào đạt: Cover là tầng CUỐI, nhận nhầm ở đây ra thẳng kết luận về quyền.
  - Máy hết RAM khiến tiến trình nền bị dừng 3 lần (IDE 2,4 GB + Chrome 2 GB + WSL + Docker). Quy tắc rút ra: **chỉ chạy một tiến trình nặng một lúc**; embedding MERT chiếm ~1,2 GB.
  - `.gitignore` thêm cặp `mert_faiss.index`/`faiss_id_map.json` (~150 MB, vượt giới hạn GitHub) và `cover_descriptors.npy`/`cover_id_map.json` (~75 MB) — dựng lại được từ audio.
