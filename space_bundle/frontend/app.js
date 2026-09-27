/* Music Rights AI — logic 4 màn hình MVP (CLAUDE.md §13).
 *
 * Nguyên tắc bám theo §2 khi hiển thị:
 *   - Không bịa tiến trình. Danh sách bước ở màn Processing được tô sáng theo
 *     trường `stage` THẬT mà backend ghi vào bảng jobs, không phải theo timer.
 *   - Không gộp ba loại độ tin cậy thành một con số.
 *   - UNKNOWN được hiển thị đúng là UNKNOWN, không làm tròn thành LOW/HIGH.
 *   - Lỗi hiện mã chuẩn hoá của §12, không hiện traceback.
 */
'use strict';

const API = '/api/v1';
const POLL_MS = 800;
const WARMUP_POLL_MS = 5000;
// Máy chủ hạn chế / không trả lời: hỏi lại /health định kỳ để trang tự mở khoá khi server
// được sửa xong, không bắt người dùng tải lại trang
const HEALTH_RETRY_MS = 10000;

// Thứ tự phải khớp <li data-stage> trong index.html
const STAGE_ORDER = [
  'VALIDATING', 'EXTRACTING_AUDIO', 'FINGERPRINTING',
  'EMBEDDING', 'VECTOR_SEARCH', 'COVER_SEARCH', 'RIGHTS_LOOKUP', 'RULE_ENGINE',
  'REGISTRY_LOOKUP',
];

// Bốn màn của luồng Phân tích; màn còn lại ('registry') thuộc tab Thống kê & Tra cứu
const ANALYZE_SCREENS = ['upload', 'processing', 'result', 'evidence'];

// URL của từng màn Phân tích, để Back/Forward đi đúng giữa Tải lên ↔ Kết quả ↔ Bằng chứng.
// KHÔNG chứa job_id (khoá truy cập, docs/SECURITY.md §1): tải lại trang thì kết quả không
// còn trong bộ nhớ và route() đưa về màn Tải lên.
const ANALYZE_HASH = {
  upload: '#phan-tich', processing: '#phan-tich',
  result: '#phan-tich/ket-qua', evidence: '#phan-tich/bang-chung',
};
const ANALYZE_ROUTE = /^#phan-tich(?:\/(ket-qua|bang-chung))?$/;
const ANALYZE_SUBROUTE = { 'ket-qua': 'result', 'bang-chung': 'evidence' };
const HOME_HASH = '#trang-chu';
const HISTORY_HASH = '#lich-su';

// Bước nào của pipeline ứng với con số latency nào trong evidence
const STAGE_TIMING = {
  FINGERPRINTING: 'fingerprint_ms',
  EMBEDDING: 'embedding_ms',
  VECTOR_SEARCH: 'vector_search_ms',
  COVER_SEARCH: 'cover_ms',
};

const RISK_EMOJI = { LOW: '🟢', CONDITIONAL: '🟡', HIGH: '🔴', UNKNOWN: '⚪' };
const RISK_TEXT = {
  LOW: 'THẤP', CONDITIONAL: 'CÓ ĐIỀU KIỆN', HIGH: 'CAO', UNKNOWN: 'CHƯA XÁC ĐỊNH',
};

// Khớp audio_service.VIDEO_EXTENSIONS: các đuôi này phải qua FFmpeg để tách tiếng
const VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.mov', '.webm', '.avi', '.flv', '.wmv', '.m4v'];
// Khớp audio_service.FFMPEG_AUDIO_EXTENSIONS: libsndfile không giải mã được AAC
const FFMPEG_AUDIO_EXTENSIONS = ['.m4a', '.aac'];

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ffmpegReady: null = chưa biết (health chưa về / lỗi), true/false theo /health
// serverBlocked: lý do máy chủ chắc chắn từ chối mọi file (CSDL hỏng…), hoặc null
// gateMessage: câu chặn đang hiện ở #upload-error, để biết lúc nào được gỡ đi
// view: tab đang mở. lastAnalyzeScreen: màn Phân tích cần quay về khi đổi tab.
const state = {
  file: null, jobId: null, result: null, timer: null, ffmpegReady: null,
  serverBlocked: null, gateMessage: null,
  view: 'home', lastAnalyzeScreen: 'upload', polling: false, viaTab: false,
};

/* ─────────────────────────── Tiện ích ─────────────────────────── */

// Người dùng bật "giảm chuyển động" trong hệ điều hành thì cuộn nhảy thẳng, không trượt
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const scrollBehavior = () => (reducedMotion() ? 'auto' : 'smooth');

function showScreen(name) {
  $$('.screen').forEach((s) => s.classList.toggle('active', s.id === `screen-${name}`));
  const inAnalyze = ANALYZE_SCREENS.includes(name);
  state.view = inAnalyze ? 'analyze' : name;   // 'home' | 'history' | 'registry'
  if (inAnalyze) state.lastAnalyzeScreen = name;
  // Thanh 4 bước là tiến trình của luồng Phân tích, không có nghĩa ở tab Thống kê
  $('.steps').hidden = !inAnalyze;
  $$('.tab').forEach((tab) => {
    const active = tab.dataset.view === state.view;
    tab.classList.toggle('active', active);
    if (active) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
  });
  const reached = ANALYZE_SCREENS.indexOf(name);
  $$('.steps li').forEach((li, i) => {
    li.classList.toggle('active', i === reached);
    li.classList.toggle('done', i < reached);
  });
  if (inAnalyze) updateSteps(name);
  window.scrollTo({ top: 0, behavior: scrollBehavior() });
}

/** Bước nào bấm được. Đang theo dõi job thì chỉ nút "Huỷ theo dõi" rời màn Xử lý;
 *  Kết quả / Bằng chứng cần có kết quả; "Xử lý" là trạng thái, không phải nơi đến. */
function stepReachable(step, current) {
  if (step === current || step === 'processing' || state.polling) return false;
  return step === 'upload' || Boolean(state.result);
}

function updateSteps(current) {
  $$('.steps .step').forEach((button) => {
    const { step } = button.dataset;
    button.disabled = !stepReachable(step, current);
    if (step === current) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
}

/** Chuyển màn do việc chạy nền (job xong / lỗi): người dùng đang xem tab Thống kê
 *  thì không giật họ về — chỉ nhớ màn đó để lần bấm tab Phân tích mở đúng chỗ. */
function showAnalyzeScreen(name, options) {
  if (state.view !== 'analyze') {
    state.lastAnalyzeScreen = name;
    return;
  }
  goAnalyze(name, options);
}

/** Focus vào heading của màn vừa mở: trình đọc màn hình đọc tên màn, phím Tab tiếp theo
 *  đi từ đầu màn — không kẹt ở nút vừa bị ẩn cùng màn cũ. */
function focusScreenHeading(name) {
  const heading = $(`#screen-${name} h2`);
  if (heading) heading.focus({ preventScroll: true });
}

/** Mở một màn của luồng Phân tích. push=true thêm một mục lịch sử để Back quay lại được. */
function goAnalyze(name, { push = true, focus = true } = {}) {
  const hash = ANALYZE_HASH[name];
  if (push && location.hash !== hash) history.pushState(null, '', hash);
  else replaceHash(hash);
  showScreen(name);
  if (focus) focusScreenHeading(name);
}

function humanSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** Trả về text hiển thị cho giá trị có thể là null/bool — KHÔNG bịa mặc định. */
// Tên bài và nghệ sĩ là văn bản tự do lấy từ dataset ngoài (FMA, Jamendo), nên có
// thể chứa < > & " '. Bảng top-K dựng bằng innerHTML, chèn thẳng vào đó là mở
// đường cho mã lạ chạy trong trang.
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

// Chỉ nhận liên kết http(s). Một giá trị kiểu "javascript:..." trong dữ liệu nguồn
// sẽ chạy mã khi người dùng bấm vào nếu đưa thẳng vào href.
function safeHttpUrl(value) {
  try {
    const url = new URL(String(value));
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

function show(value, { yes = 'Có', no = 'Không', empty = '—' } = {}) {
  if (value === null || value === undefined || value === '') return empty;
  if (value === true) return yes;
  if (value === false) return no;
  return String(value);
}

function dl(container, rows) {
  container.innerHTML = '';
  rows.forEach(([label, value, opts = {}]) => {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    const classes = [opts.mono ? 'mono' : '', opts.className || ''].filter(Boolean);
    if (classes.length) dd.className = classes.join(' ');

    if (opts.html) {
      dd.innerHTML = opts.html;
    } else if ((value === true || value === false) && opts.neutral) {
      // Có/Không không mang nghĩa tốt/xấu (vd. "bắt buộc ghi nguồn: Có" là một điều
      // kiện, "PD của bản thu: Không" là một trạng thái) — không tô xanh/đỏ
      dd.textContent = show(value, opts);
    } else if (value === true || value === false) {
      dd.innerHTML = `<span class="${value ? 'yes' : 'no'}">${show(value, opts)}</span>`;
    } else {
      dd.textContent = show(value, opts);
    }
    container.append(dt, dd);
  });
}

/** Đuôi file viết thường, kể cả dấu chấm ('.mp3'); không có đuôi thì ''. */
function fileExt(name) {
  const dot = name.lastIndexOf('.');
  return dot >= 0 ? name.slice(dot).toLowerCase() : '';
}

/** fetch() ném TypeError khi không tới được máy chủ ("Failed to fetch"): đổi thành câu
 *  người dùng hiểu được. Lỗi có mã §12 thì giữ nguyên. */
function describeError(err) {
  if (err instanceof TypeError) {
    return 'Không kết nối được máy chủ. Kiểm tra server backend đang chạy rồi thử lại.';
  }
  return err.message;
}

/** Đọc lỗi API: FastAPI bọc mã lỗi §12 trong `detail`. */
async function apiError(response) {
  let detail;
  try {
    detail = (await response.json()).detail;
  } catch (_) {
    return `Lỗi HTTP ${response.status}`;
  }
  if (detail && detail.error_code) return `[${detail.error_code}] ${detail.message}`;
  return typeof detail === 'string' ? detail : `Lỗi HTTP ${response.status}`;
}

/** fetch + kiểm tra response.ok + đọc mã lỗi §12 — dùng cho các lệnh gọi của sổ. */
async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await apiError(response));
  return response.json();
}

/* ─────────────────────────── Health ─────────────────────────── */

const WARMUP_STATUS = { RUNNING: 'đang chạy', DONE: 'xong', DISABLED: 'tắt' };

function describeWarmup(warmup) {
  if (!warmup || !warmup.status) return 'không rõ';
  if (warmup.status === 'DISABLED') return 'tắt — truy vấn đầu tiên sẽ chậm hơn';
  const steps = Object.entries(warmup.steps || {}).map(([name, step]) =>
    `${name} ${step.status}${step.seconds !== undefined ? ` ${step.seconds} s` : ''}`);
  return `${WARMUP_STATUS[warmup.status] || warmup.status}`
    + (steps.length ? ` (${steps.join(', ')})` : '');
}

/** Hộp thông báo ở màn Tải lên: máy chủ chặn hẳn (blocking) hoặc chỉ hạn chế. */
function renderServerNote(blocking, issues) {
  const note = $('#server-note');
  note.hidden = !issues.length;
  note.classList.toggle('blocking', Boolean(blocking));
  if (!issues.length) {
    note.replaceChildren();
    return;
  }
  const head = document.createElement('p');
  head.textContent = blocking ? `Chưa phân tích được file: ${blocking}`
    : 'Máy chủ đang chạy ở chế độ hạn chế:';
  const list = document.createElement('ul');
  list.append(...issues.map((issue) => {
    const li = document.createElement('li');
    li.textContent = issue;
    return li;
  }));
  note.replaceChildren(head, list);
}

async function checkHealth() {
  const box = $('#health');
  const dot = box.querySelector('.dot');
  const text = box.querySelector('.health-text');
  try {
    const health = await (await fetch('/health')).json();
    const c = health.components;
    const bits = [
      `DB ${c.database}`,
      `FAISS ${c.faiss.status} (${c.faiss.total_recordings} bản ghi)`,
      `Chromaprint ${c.chromaprint.status}`,
      // Thiếu FFmpeg thì file video (.mp4/.mov) không tách được tiếng — người dùng
      // cần biết TRƯỚC khi tải lên, không phải sau khi nhận lỗi
      `FFmpeg ${c.ffmpeg === 'AVAILABLE' ? 'AVAILABLE' : 'MISSING — chỉ nhận file audio'}`,
      ...(c.cover ? [`Cover ${c.cover.status}`] : []),
      `Rule Engine ${c.rule_engine.version || c.rule_engine.status}`,
      ...(health.thresholds ? [`τFP=${health.thresholds.fingerprint} τMERT=${health.thresholds.embedding} `
        + `τCover=${health.thresholds.cover}`] : []),
      `Làm nóng: ${describeWarmup(health.warmup)}`,
    ];
    // Máy chủ vừa bật: vẫn nhận file, nhưng truy vấn gửi lúc này chờ bộ đệm nạp xong
    // (~25 s) — nói trước để người dùng không tưởng hệ thống treo ở bước Chromaprint.
    const warming = Boolean(health.warmup) && health.warmup.status === 'RUNNING';
    const online = health.status === 'ONLINE';
    dot.className = `dot ${online && !warming ? 'dot-ok' : 'dot-warn'}`;
    text.textContent = !online ? 'Hoạt động hạn chế'
      : (warming ? 'Sẵn sàng · đang nạp bộ đệm' : 'Hệ thống sẵn sàng');
    $('#health-details').replaceChildren(...bits.map((bit) => {
      const li = document.createElement('li');
      li.textContent = bit;
      return li;
    }));
    const indexReady = c.faiss.status === 'READY';
    $('#hero-meta').innerHTML = `<span class="dot ${online ? 'dot-ok' : 'dot-warn'}" aria-hidden="true"></span>`
      + (indexReady ? `Đang đối chiếu với ${fmtNumber(c.faiss.total_recordings)} bản ghi`
        : 'Chỉ mục MERT chưa sẵn sàng')
      + ` · Rule Engine ${escapeHtml(c.rule_engine.version || '—')}`;

    // Thiếu CSDL thì /analyze không tạo được job, thiếu Rule Engine thì không ra được kết
    // luận: mọi file đều hỏng — chặn gửi và nói lý do TRƯỚC khi người dùng chờ tải lên
    const dbReady = c.database === 'CONNECTED';
    const rulesReady = c.rule_engine.status === 'READY';
    state.serverBlocked = !dbReady ? 'máy chủ chưa kết nối được cơ sở dữ liệu.'
      : (!rulesReady ? 'máy chủ chưa nạp được Rule Engine.' : null);
    state.ffmpegReady = c.ffmpeg === 'AVAILABLE';
    const issues = [];
    if (!dbReady) {
      issues.push('Cơ sở dữ liệu: chưa kết nối được. Quản trị viên kiểm tra DATABASE_URL '
        + 'trong file .env và container PostgreSQL, rồi khởi động lại server.');
    }
    if (!rulesReady) issues.push('Rule Engine: không nạp được configs/rules_v1.yaml.');
    if (!indexReady) {
      issues.push('Chỉ mục MERT chưa sẵn sàng: chỉ nhận ra bản ghi gần như giống hệt, bài khác '
        + 'sẽ báo MODEL_FAILURE. Quản trị viên chạy python scripts/rebuild_faiss_index.py --from-db.');
    }
    if (!state.ffmpegReady) issues.push('Không có FFmpeg: chưa nhận file video và m4a/aac.');
    renderServerNote(state.serverBlocked, issues);
    refreshUploadGate();   // file chọn trước khi health về: xét lại, không nạp lại file

    if (warming) setTimeout(checkHealth, WARMUP_POLL_MS);
    else if (!online) setTimeout(checkHealth, HEALTH_RETRY_MS);
  } catch (_) {
    dot.className = 'dot dot-bad';
    text.textContent = 'Không kết nối được máy chủ';
    $('#hero-meta').innerHTML = '<span class="dot dot-bad" aria-hidden="true"></span>Chưa kết nối được máy chủ';
    const li = document.createElement('li');
    li.textContent = 'GET /health không trả lời — kiểm tra server backend.';
    $('#health-details').replaceChildren(li);
    // Không chặn: có thể chỉ là mạng chập chờn, lượt gửi sẽ tự báo lỗi nếu vẫn không tới
    renderServerNote(null, ['Không kết nối được máy chủ — kiểm tra server backend đang chạy.']);
    setTimeout(checkHealth, HEALTH_RETRY_MS);
  }
}

/* ─────────────────────────── Màn 1: Upload ─────────────────────────── */

/** Lý do chưa gửi được file (chuỗi) hoặc null. Máy chủ hỏng CSDL, hoặc thiếu FFmpeg với
 *  video / m4a / aac, thì file chắc chắn bị từ chối — báo ngay khi chọn file thay vì sau
 *  khi tải lên. Chưa biết trạng thái máy chủ (health lỗi) thì không chặn. */
function uploadBlockedReason(file) {
  if (!file) return null;
  if (state.serverBlocked) return `Chưa gửi được: ${state.serverBlocked} Xem thông báo phía trên.`;
  if (state.ffmpegReady !== false) return null;
  const ext = fileExt(file.name);
  if (VIDEO_EXTENSIONS.includes(ext)) {
    return 'Máy chủ hiện không có FFmpeg nên chưa tách được âm thanh từ video. '
      + 'Hãy tải lên file audio (.mp3, .wav, .flac…).';
  }
  if (FFMPEG_AUDIO_EXTENSIONS.includes(ext)) {
    return `Máy chủ hiện không có FFmpeg nên chưa giải mã được file ${ext}. `
      + 'Hãy đổi sang .mp3, .wav hoặc .flac rồi tải lên.';
  }
  return null;
}

/** Bật/tắt nút PHÂN TÍCH và câu chặn theo file đang chọn + trạng thái máy chủ. Gọi lại được
 *  bất cứ lúc nào (mỗi lần /health về) mà không nạp lại file, không xoá lỗi của job trước. */
function refreshUploadGate() {
  const blocked = uploadBlockedReason(state.file);
  const button = $('#analyze');
  // Đang tải lên: nút phải giữ trạng thái khoá, kẻo bấm được lần hai
  if (button.getAttribute('aria-busy') !== 'true') button.disabled = !state.file || Boolean(blocked);
  const box = $('#upload-error');
  if (blocked) {
    box.textContent = blocked;
    box.hidden = false;
  } else if (state.gateMessage && box.textContent === state.gateMessage) {
    box.textContent = '';
    box.hidden = true;
  }
  state.gateMessage = blocked;
}

function pickFile(file) {
  state.file = file || null;
  $('#file-picked').hidden = !file;
  // File mới: lỗi của lần gửi trước không còn đúng nữa
  $('#upload-error').textContent = '';
  $('#upload-error').hidden = true;
  state.gateMessage = null;
  refreshUploadGate();
  if (file) {
    $('#file-name').textContent = file.name;
    $('#file-size').textContent = humanSize(file.size);
    // Nảy nhẹ khi nhận file: gỡ rồi gắn lại class để chạy lại animation
    const picked = $('#file-picked');
    picked.classList.remove('received');
    void picked.offsetWidth;
    picked.classList.add('received');
  }
  prepareWave(state.file);
}

function initUpload() {
  const dropzone = $('#dropzone');
  const input = $('#file');

  const open = () => input.click();
  dropzone.addEventListener('click', (e) => { if (e.target !== input) open(); });
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
  });
  $('#browse').addEventListener('click', (e) => { e.stopPropagation(); open(); });
  input.addEventListener('change', () => pickFile(input.files[0]));

  ['dragenter', 'dragover'].forEach((type) =>
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragging');
    }));
  ['dragleave', 'drop'].forEach((type) =>
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragging');
    }));
  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      pickFile(e.dataTransfer.files[0]);
    }
  });

  $('#clear-file').addEventListener('click', (e) => {
    e.stopPropagation();
    input.value = '';
    pickFile(null);
  });

  $('#upload-form').addEventListener('submit', submitAnalysis);
}

async function submitAnalysis(event) {
  event.preventDefault();
  if (!state.file || uploadBlockedReason(state.file)) return;

  const body = new FormData();
  body.append('file', state.file);
  body.append('platform', $('#platform').value);
  body.append('commercial_use', $('#commercial_use').value);
  body.append('monetization', $('#monetization').value);

  const button = $('#analyze');
  button.disabled = true;
  // File tới 100 MB có thể mất vài giây để tải lên — nút phải báo là đang làm
  button.textContent = 'ĐANG TẢI LÊN…';
  button.setAttribute('aria-busy', 'true');
  $('#upload-error').hidden = true;

  try {
    const response = await fetch(`${API}/analyze`, { method: 'POST', body });
    if (!response.ok) throw new Error(await apiError(response));

    const { job_id } = await response.json();
    state.jobId = job_id;
    state.result = null;     // kết quả cũ thuộc file trước: bước 3/4 không được mở nó nữa
    state.polling = true;
    $('#job-id').textContent = job_id;
    $('#processing-file').textContent = `· ${state.file.name}`;
    resetStages();
    goAnalyze('processing', { push: false });
    pollJob();
  } catch (err) {
    const box = $('#upload-error');
    box.textContent = describeError(err);
    box.hidden = false;
  } finally {
    button.textContent = 'PHÂN TÍCH';
    button.removeAttribute('aria-busy');
    button.disabled = !state.file || Boolean(uploadBlockedReason(state.file));
  }
}

/* ─────────────────────────── Màn 2: Processing ─────────────────────────── */

function resetStages() {
  $$('#stage-list li').forEach((li) => {
    li.className = '';
    li.removeAttribute('aria-current');
    const ms = li.querySelector('.ms');
    if (ms) ms.remove();
  });
  $('#job-status').textContent = 'QUEUED';
  $('#stage-live').textContent = '';
  paintProgress(0, STAGE_ORDER.length, 'Chưa bắt đầu');
}

/** Tên bước (nút văn bản đầu của <li>, không lẫn nhãn độ trễ thêm vào sau). */
const stageLabel = (li) => li.firstChild.textContent.trim();

/** Tô danh sách bước theo `stage` THẬT do backend báo. */
function paintStages(stage) {
  const current = STAGE_ORDER.indexOf(stage);
  const items = $$('#stage-list li');
  items.forEach((li, i) => {
    if (current < 0) { li.className = ''; li.removeAttribute('aria-current'); return; }
    li.className = i < current ? 'done' : (i === current ? 'running' : '');
    if (i === current) li.setAttribute('aria-current', 'step');
    else li.removeAttribute('aria-current');
  });
  // Chỉ ghi khi bước đổi: poll 800 ms/lần, ghi lại cùng câu là đọc lặp
  const live = $('#stage-live');
  const text = current >= 0
    ? `Bước ${current + 1}/${STAGE_ORDER.length}: ${stageLabel(items[current])}` : '';
  if (live.textContent !== text) live.textContent = text;
  if (current >= 0) paintProgress(current + 1, STAGE_ORDER.length, text);
}

/** Kết thúc job: bước nào CHẠY THẬT thì đánh dấu xong kèm độ trễ đo được, bước
 *  nào bị bỏ qua thì nói rõ là bỏ qua — không tô "xong" cho việc chưa hề làm. */
function markAllStagesDone(result) {
  $('#stage-live').textContent = 'Phân tích xong — đang mở kết quả.';
  paintProgress(STAGE_ORDER.length, STAGE_ORDER.length, 'Xong');
  const identification = (result.evidence || {}).identification || {};
  const timings = identification.timings_ms || {};
  const stoppedAtStage1 = (result.match || {}).pipeline_stage === 'STAGE_1_CHROMAPRINT';
  const identified = Boolean((result.identity || {}).recording_id);

  $$('#stage-list li').forEach((li) => {
    const stage = li.dataset.stage;
    const key = STAGE_TIMING[stage];
    let value = key ? timings[key] : undefined;

    let skipped = false;
    let note = '';
    if (key && (value === undefined || value === null)) {
      skipped = true;
      if (stage === 'COVER_SEARCH') {
        note = stoppedAtStage1 ? 'bỏ qua — tầng 1 đã khớp' : 'bỏ qua — tầng 2 đã khớp';
      } else {
        note = stoppedAtStage1 ? 'bỏ qua — tầng 1 đã khớp' : 'không chạy';
      }
    } else if (stage === 'RIGHTS_LOOKUP' && !identified) {
      skipped = true;
      note = 'bỏ qua — chưa định danh được bản ghi';
    } else if (stage === 'REGISTRY_LOOKUP') {
      const registry = (result.evidence || {}).unknown_registry;
      if (registry && !registry.error_code) {
        value = registry.elapsed_ms;
      } else {
        skipped = true;
        note = registry ? 'không ghi được sổ'
          : (isUnknownResult(result) ? 'không chạy' : 'bỏ qua — kết quả đã xác định');
      }
    }

    li.className = skipped ? 'skipped' : 'done';
    li.removeAttribute('aria-current');

    const label = document.createElement('span');
    label.className = 'ms';
    label.textContent = skipped ? note : (value !== undefined && value !== null
      ? `${Math.round(value)} ms` : '');
    const existing = li.querySelector('.ms');
    if (existing) existing.remove();
    if (label.textContent) li.append(label);
  });
}

function stopPolling() {
  if (state.timer) { clearTimeout(state.timer); state.timer = null; }
  state.polling = false;
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const response = await fetch(`${API}/jobs/${state.jobId}`);
    if (!response.ok) throw new Error(await apiError(response));
    const job = await response.json();

    $('#job-status').textContent = job.status;
    paintStages(job.stage);

    if (job.status === 'DONE') return loadResult();
    if (job.status === 'FAILED') {
      stopPolling();
      showAnalyzeScreen('upload', { push: false });
      const box = $('#upload-error');
      box.textContent = `[${job.error_code}] ${job.error_message}`;
      box.hidden = false;
      return;
    }
    state.timer = setTimeout(pollJob, POLL_MS);
  } catch (err) {
    stopPolling();
    showAnalyzeScreen('upload', { push: false });
    const box = $('#upload-error');
    box.textContent = describeError(err);
    box.hidden = false;
  }
}

async function loadResult() {
  stopPolling();
  try {
    const response = await fetch(`${API}/results/${state.jobId}`);
    if (!response.ok) throw new Error(await apiError(response));
    state.result = await response.json();
    markAllStagesDone(state.result);
    renderResult(state.result);
    renderEvidence(state.result);
    recordHistory(state.jobId, state.result, state.file ? state.file.name : null);
    showAnalyzeScreen('result');
  } catch (err) {
    // Lỗi mạng, JSON hỏng hay lỗi khi dựng màn hình: trước đây không được bắt nên
    // màn Processing đứng yên mãi, người dùng không biết chuyện gì xảy ra.
    const box = $('#upload-error');
    box.textContent = describeError(err) || 'Không tải được kết quả phân tích.';
    box.hidden = false;
    showAnalyzeScreen('upload', { push: false });
  }
}

/* ─────────────────────────── Màn 3: Result ─────────────────────────── */

/** Giá trị quyền do mô hình suy đoán: thành chữ kèm "(suy đoán)" — không còn là
 *  true/false nên dl() không tô xanh/đỏ. Quyền tra được thì giữ nguyên. */
function asGuess(value, guessed) {
  if (!guessed || value === null || value === undefined || value === '') return value;
  return `${show(value)} (suy đoán)`;
}

function renderResult(result) {
  const { identity = {}, match = {}, rights = {}, assessment = {} } = result;
  const risk = assessment.risk || 'UNKNOWN';
  const identification = (result.evidence || {}).identification || {};
  const identified = Boolean(identity.recording_id);

  const banner = $('#risk-banner');
  banner.className = `risk-banner risk-${risk}`;
  $('#risk-emoji').textContent = RISK_EMOJI[risk] || '⚪';
  $('#risk-level').textContent = `${risk} — ${RISK_TEXT[risk] || ''}`;
  // Focus vào heading này khi kết quả về: trình đọc màn hình đọc ngay mức rủi ro
  $('#result-title').textContent = `Kết quả phân tích: mức rủi ro ${risk} — ${RISK_TEXT[risk] || ''}`;
  // Độ tin cậy nhận diện của backend. Chưa định danh được thì đây là điểm cao nhất DƯỚI
  // ngưỡng — ghi rõ, không để con số trông như "khớp 5%".
  const percent = match.confidence == null ? NaN : Number(match.confidence) * 100;
  countUp($('#match-percent'), percent);
  $('#match-percent-sr').textContent = Number.isFinite(percent)
    ? `${percent.toLocaleString('vi-VN', { maximumFractionDigits: 1 })}%` : 'không có';
  $('#match-label').textContent = identified ? 'Độ tin cậy nhận diện'
    : 'Điểm cao nhất — dưới ngưỡng, chưa định danh được';
  $('#risk-condition').textContent = assessment.condition
    ? `Điều kiện: ${assessment.condition}` : '';

  // Đánh giá ngữ cảnh xấu nhất (Worst-Case Context)
  const worstCase = assessment.worst_case;
  const wcBanner = $('#worst-case-banner');
  const wcText = $('#worst-case-text');
  const wcNote = $('#worst-case-note');

  if (wcBanner && worstCase && (worstCase.risk !== risk || worstCase.condition !== assessment.condition)) {
    wcBanner.hidden = false;
    wcBanner.className = `worst-case-banner worst-case-${worstCase.risk || 'CONDITIONAL'}`;
    const condText = worstCase.condition ? ` — ${worstCase.condition}` : '';
    wcText.textContent = `Nếu dùng thương mại và bật kiếm tiền: ${worstCase.risk || ''}${condText}`;
    if (assessment.rights_source_note) {
      wcNote.textContent = assessment.rights_source_note;
      wcNote.hidden = false;
    } else {
      wcNote.textContent = '';
      wcNote.hidden = true;
    }
  } else if (wcBanner) {
    wcBanner.hidden = true;
    if (wcText) wcText.textContent = '';
    if (wcNote) wcNote.textContent = '';
  }

  dl($('#identity-list'), [
    ['Bài hát', identity.track],
    ['Nghệ sĩ', identity.artist],
    ['Loại khớp', null, {
      html: `<span class="pill">${show(match.type)}</span>`,
    }],
    ['Độ tin cậy nhận diện', match.confidence !== undefined && match.confidence !== null
      ? match.confidence.toFixed(4) : null],
    // Điểm Chromaprint, cosine MERT và điểm Cover KHÔNG cùng thang: 0.938 là "không
    // đạt" với MERT (τ 0.98) nhưng "đạt" với Cover (τ 0.90). Câu của chính cascade
    // nêu điểm từng tầng so với ngưỡng của tầng đó, nên con số trên đọc được đúng.
    ['Căn cứ nhận diện', identification.decision_reason],
    ['Tầng xử lý', match.pipeline_stage],
    ['recording_id', identity.recording_id, { mono: true }],
    ['composition_id', identity.composition_id, { mono: true }],
  ]);

  // Quyền SUY ĐOÁN từ âm thanh và quyền TRA CỨU phải trình bày khác hẳn nhau (§2.4).
  // Một dòng nhãn là chưa đủ: các ô Có/Không vẫn tô xanh/đỏ y như quyền tra được, cho
  // con số đoán vẻ chắc chắn mà EXP-09 không ủng hộ (dưới baseline lớp phổ biến).
  const guessed = Boolean(rights.predicted);
  $('#card-rights').classList.toggle('predicted', guessed);
  dl($('#rights-list'), [
    ['Nguồn gốc giấy phép', guessed
      ? 'SUY ĐOÁN bởi mô hình từ âm thanh — không phải tra cứu, cần người kiểm tra'
      : (rights.rights_found ? 'Tra cứu từ cơ sở dữ liệu quyền' : null),
    { className: guessed ? 'guess-warning' : '' }],
    ['Giấy phép', asGuess(rights.license, guessed)],
    ['Trạng thái bản quyền', asGuess(rights.copyright_status, guessed)],
    // Ghi nguồn là một ĐIỀU KIỆN, không phải điều tốt — không tô xanh
    ['Bắt buộc ghi nguồn', asGuess(rights.attribution_required, guessed), { neutral: true }],
    ['Cho dùng thương mại', asGuess(rights.commercial_use_allowed, guessed)],
    ['Cho bật kiếm tiền', asGuess(rights.monetization_allowed, guessed)],
    ['Nguồn dữ liệu', rights.source],
    ['Ghi chú nguồn quyền', assessment.rights_source_note],
    ['Xác minh lần cuối', rights.verified_at],
  ]);

  renderRegistryCard(result);
  staggerReveal();

  $('#recommendation').textContent = result.recommendation || '—';
  $('#decision-reason').textContent = assessment.reason
    ? `Lý do quyết định: ${assessment.reason}` : '';

  // Chưa định danh được thì identity_confidence là điểm cao nhất DƯỚI ngưỡng (backend
  // cố ý giữ điểm thật, §2.3). Thanh đầy 94% cùng màu với một khớp thật sẽ đọc thành
  // "khá chắc" — làm mờ và ghi rõ, không giấu con số.
  const confidences = [
    [identified ? 'Độ tin cậy nhận diện' : 'Độ tin cậy nhận diện (chưa định danh được)',
      assessment.identity_confidence, !identified],
    ['Độ tin cậy dữ liệu quyền', assessment.rights_confidence, false],
    ['Độ tin cậy quyết định', assessment.decision_confidence, false],
  ];
  $('#confidences').innerHTML = confidences.map(([label, value, below]) => {
    const v = Number(value || 0);
    return `<div class="conf-row"><span>${label}</span>
      <span class="bar${below ? ' below' : ''}"><i style="width:${(v * 100).toFixed(0)}%"></i></span>
      <span class="mono">${v.toFixed(3)}</span></div>`;
  }).join('');

  $('#feedback-msg').hidden = true;
  $$('.feedback-buttons button').forEach((b) => b.setAttribute('aria-pressed', 'false'));
}

/* ─────────────────────────── Màn 4: Evidence ─────────────────────────── */

const FULL_SCAN_REASON = {
  NEAR_THRESHOLD: 'điểm sau lọc nằm sát ngưỡng',
  QUERY_TOO_SHORT: 'truy vấn quá ngắn để lọc theo hash',
  PREFILTER_DISABLED: 'bộ lọc đang tắt',
};

function describePrefilter(prefilter) {
  if (!prefilter) return null;
  if (prefilter.full_scan) {
    const why = FULL_SCAN_REASON[prefilter.full_scan_reason] || prefilter.full_scan_reason;
    return `Quét toàn bộ bảng fingerprint (${why})`;
  }
  return `Lọc theo hash trùng: chấm đầy đủ ${prefilter.candidates} bản ghi có nhiều hash `
    + `trùng nhất (tối đa ${prefilter.top_k}); điểm cách xa ngưỡng nên không cần quét toàn bộ`;
}

/** Giống cover_service.describe_oti: OTI là số bán cung phải dịch truy vấn LÊN để
 *  khớp, nên "OTI 11" là truy vấn CAO hơn bản gốc 1 bán cung — hiện "11 bán cung"
 *  trần thì đọc thành lệch gần một quãng tám. */
function describeOti(oti) {
  if (oti === undefined || oti === null) return null;
  const k = ((Number(oti) % 12) + 12) % 12;
  if (k === 0) return 'OTI 0: cùng cao độ với bản gốc';
  const shift = (12 - k) % 12;
  if (shift === 6) return `OTI ${k}: lệch nửa quãng tám (6 bán cung, chroma không phân biệt được chiều)`;
  const signed = shift > 6 ? shift - 12 : shift;
  return `OTI ${k}: truy vấn ${signed > 0 ? 'cao' : 'thấp'} hơn bản gốc ${Math.abs(signed)} bán cung`;
}

/** Hệ số nhịp độ của đoạn cắt đã thắng ở tầng Cover (lưới COVER_TEMPO_FACTORS, không
 *  phải nhịp đo được) — 0.9 nghĩa là truy vấn chậm hơn bản gốc khoảng 10%.
 *  null: truy vấn ngắn hơn độ dài cần cắt nên đoạn cắt bị cụt, không gắn được hệ số. */
function describeTempo(factor) {
  if (factor === undefined) return null;
  if (factor === null) return 'không suy ra được (truy vấn ngắn hơn độ dài cần cắt)';
  if (Number(factor) === 1) return '×1.00 (đúng nhịp bản gốc)';
  return `×${Number(factor).toFixed(2)} (truy vấn ${Number(factor) < 1 ? 'chậm' : 'nhanh'} hơn bản gốc)`;
}

function renderEvidence(result) {
  const evidence = result.evidence || {};
  const identification = evidence.identification || {};
  const fingerprint = identification.fingerprint || {};
  const embedding = identification.embedding || {};
  const rules = evidence.rule_engine || {};
  const rightsRecord = evidence.rights_record || {};
  const composition = evidence.composition || {};

  // "Không so được" và "so rồi nhưng không khớp" là hai kết luận khác hẳn nhau;
  // hiện điểm 0.0000 cho trường hợp đầu sẽ khiến người đọc tưởng là trường hợp sau.
  const fpRan = Boolean(fingerprint.match_type) && fingerprint.match_type !== 'UNAVAILABLE';
  dl($('#ev-fingerprint'), [
    ['Điểm Chromaprint', !fpRan
      ? 'không tính — tầng 1 không chạy'
      : (fingerprint.fingerprint_score != null
        ? Number(fingerprint.fingerprint_score).toFixed(4) : null)],
    // Truy vấn ngắn dùng ngưỡng hiệu dụng cao hơn τFP (điểm nền của nhiễu tăng khi
    // đoạn ngắn đi) — không ghi rõ thì con số này lệch với τFP ở thanh trạng thái.
    ['Ngưỡng τFP', fingerprint.threshold == null ? null
      : (fingerprint.threshold_raised_for_short_query
        ? `${Number(fingerprint.threshold).toFixed(4)} (nâng từ ${fingerprint.threshold_base} vì truy vấn ngắn)`
        : fingerprint.threshold)],
    ['Kết luận tầng 1', fingerprint.match_type],
    ...(fpRan ? [] : [['Lý do tầng 1 không chạy',
      [fingerprint.reason_code, fingerprint.message].filter(Boolean).join(' — ') || null]]),
    // Tầng 1 chỉ chấm đầy đủ vài chục bản ghi nhiều hash trùng nhất; hiện trần
    // "đã so: 6" mà không nói cách dò thì người đọc tưởng CSDL chỉ có 6 fingerprint.
    ...(fpRan ? [['Cách dò', describePrefilter(fingerprint.prefilter)]] : []),
    ...(fpRan && fingerprint.prefilter && fingerprint.prefilter.best_hash_hits !== undefined
      ? [['Số hash trùng nhiều nhất (một bản ghi)', fingerprint.prefilter.best_hash_hits]] : []),
    ['Số fingerprint đã chấm đầy đủ', fingerprint.candidates_compared],
    ['Bỏ qua vì lệch độ dài', fingerprint.candidates_skipped_by_duration],
    ['Độ dài truy vấn (s)', fingerprint.query_duration
      ? Number(fingerprint.query_duration).toFixed(1) : null],
    ['Ứng viên dưới ngưỡng', fingerprint.best_candidate_below_threshold, { mono: true }],
  ]);

  const tbody = $('#ev-topk tbody');
  const candidates = embedding.top_k || evidence.candidates || [];
  tbody.innerHTML = '';
  if (!candidates.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">Tầng 2 không chạy (tầng 1 đã khớp) hoặc không có ứng viên.</td></tr>';
  } else {
    candidates.forEach((c, i) => {
      const tr = document.createElement('tr');
      if (c.recording_id === (result.identity || {}).recording_id) tr.className = 'hit';
      const segment = c.best_segment
        ? `${c.best_segment.start ?? c.best_segment[0] ?? '?'}–${c.best_segment.end ?? c.best_segment[1] ?? '?'}s`
        : '—';
      // Chỉ có UUID thì không ai đối chiếu được "0.9348 với b83646be-…" là bài nào
      const label = [c.track, c.artist].filter(Boolean).map(escapeHtml).join(' — ');
      tr.innerHTML = `<td>${i + 1}</td>
        <td>${label ? `${label}<br>` : ''}<span class="mono muted">${escapeHtml(c.recording_id)}</span></td>
        <td>${Number(c.similarity_score).toFixed(4)}</td>
        <td>${segment}</td><td>${show(c.segments_hit)}</td>`;
      tbody.append(tr);
    });
  }

  dl($('#ev-embedding'), [
    ['Ngưỡng τMERT', embedding.threshold],
    ['Phiên bản model', embedding.model_version || result.model_version],
    ['Số vector tham chiếu', embedding.reference_vectors],
    ['Số bản ghi tham chiếu', embedding.reference_recordings],
    ['Số đoạn đã dò', embedding.segments_probed],
  ]);

  dl($('#ev-rules'), [
    ['Phiên bản bộ luật', rules.rules_version],
    ['Luật kích hoạt', rules.rule_id, { mono: true }],
    ['Nhóm bản quyền', result.assessment ? result.assessment.category : null],
    ['Thứ tự ưu tiên nhóm', rules.group_order],
    // Ngưỡng tách THEO loại khớp (thang điểm mỗi tầng khác nhau); "0" trần cạnh một
    // kết quả UNKNOWN đọc như "ngưỡng 0 mà vẫn không qua" — ghi rõ nó của loại nào.
    ['Ngưỡng định danh tối thiểu', rules.min_identity_confidence == null ? null
      : `${rules.min_identity_confidence}${rules.match_type ? ` (loại khớp ${rules.match_type}`
        + (rules.match_type === 'LICENSE_PREDICTED'
          ? ' — không có định danh nào để đặt ngưỡng; chặn bằng độ tin cậy dữ liệu quyền)' : ')')
        : ''}`],
    ['Điều kiện đã khớp', rules.matched_conditions
      ? JSON.stringify(rules.matched_conditions) : null, { mono: true }],
    ['Trừ điểm dữ liệu quyền', (rules.rights_confidence_penalties || []).join('; ') || null],
    ['Phép tính độ tin cậy dữ liệu quyền', rules.rights_confidence_breakdown
      ? rules.rights_confidence_breakdown.formula : null],
    ['Phép tính độ tin cậy quyết định', rules.decision_confidence_formula],
    ...(rules.provisional_decision ? [
      ['Kết luận tạm (bị chặn vì quyền kém tin cậy)',
        [rules.provisional_decision.risk_level, rules.provisional_decision.category,
          rules.provisional_decision.condition].filter(Boolean).join(' · ')],
      ['Ngưỡng tin cậy quyền tối thiểu',
        `${rules.min_rights_confidence} (còn thiếu ${rules.rights_shortfall})`],
    ] : []),
    ['Ngữ cảnh sử dụng', rules.usage_context
      ? JSON.stringify(rules.usage_context) : null, { mono: true }],
  ]);

  const guessed = Boolean((result.rights || {}).predicted);
  $('#card-source').classList.toggle('predicted', guessed);
  dl($('#ev-source'), [
    ['Nguồn metadata quyền', rightsRecord.source],
    ['Đường dẫn nguồn', null, {
      // source_url đến từ dữ liệu nguồn ngoài: thoát ký tự và chỉ nhận http(s)
      html: safeHttpUrl(rightsRecord.source_url)
        ? `<a href="${escapeHtml(rightsRecord.source_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(rightsRecord.source_url)}</a>`
        : escapeHtml(rightsRecord.source_url || '—'),
    }],
    ['Ngày xác minh', rightsRecord.verified_at],
    ['Phạm vi quyền', rightsRecord.rights_scope],
    ['Hiệu lực', rightsRecord.valid_from
      ? `${rightsRecord.valid_from} → ${show(rightsRecord.valid_until)}` : null],
    ['Lãnh thổ / nền tảng', rightsRecord.territory
      ? `${rightsRecord.territory} / ${show(rightsRecord.platform)}` : null],
    ['Tác phẩm (composition)', composition.title],
    ['Tác giả', composition.composer],
    // §2: PD của tác phẩm và PD của bản thu là hai trường ĐỘC LẬP. "Không" ở đây là
    // một trạng thái, không phải lỗi — không tô đỏ.
    ['PD của tác phẩm', asGuess(composition.public_domain_status, guessed)],
    ['PD của bản thu', asGuess(rightsRecord.recording_public_domain, guessed), { neutral: true }],
  ]);

  const cover = identification.cover || {};
  const coverCard = $('#card-cover');
  if (coverCard) {
    const hasCover = Boolean(cover && (cover.matched || cover.top_candidate || cover.error));
    if (!hasCover) {
      dl($('#ev-cover'), [
        ['Trạng thái', 'Tầng 3 không chạy (đã khớp ở tầng trước hoặc chưa bật)'],
      ]);
    } else {
      const topCand = cover.top_candidate || (cover.candidates && cover.candidates[0]) || {};
      dl($('#ev-cover'), [
        ['Khớp Cover/Phiên bản', cover.matched ? 'ĐÃ KHỚP' : 'Không đạt ngưỡng'],
        ['Điểm tương đồng CQT', topCand.similarity_score !== undefined ? Number(topCand.similarity_score).toFixed(4) : null],
        ['Lệch cao độ (OTI)', describeOti(topCand.oti)],
        ['Hệ số nhịp của đoạn cắt thắng', describeTempo(topCand.tempo_factor)],
        ['Ngưỡng τCover', cover.threshold],
        // τCover hiệu chỉnh theo SỐ BÀI trong chỉ mục, nên thiếu con số này thì
        // điểm tương đồng ở trên không đọc được đúng
        ['Số bài đã tìm trong chỉ mục', cover.reference_recordings],
        ['Bài khớp nhất', [topCand.track, topCand.artist].filter(Boolean).join(' — ') || null],
        ['recording_id', topCand.recording_id, { mono: true }],
        ['Lỗi (nếu có)', cover.error],
      ]);
    }
  }

  const timings = identification.timings_ms || {};
  dl($('#ev-timings'), [
    ['Chromaprint', timings.fingerprint_ms ? `${timings.fingerprint_ms} ms` : null],
    ['Trích embedding MERT', timings.embedding_ms ? `${timings.embedding_ms} ms` : null],
    ['Tìm kiếm vector', timings.vector_search_ms ? `${timings.vector_search_ms} ms` : null],
    ['Nhận dạng Cover (CQT/OTI)', timings.cover_ms ? `${timings.cover_ms} ms` : null],
    ['Tổng pipeline', result.latency_ms ? `${result.latency_ms} ms` : null],
    ['Phiên bản model', result.model_version],
  ]);

  $('#ev-raw').textContent = JSON.stringify(result, null, 2);
}

/* ─────────────────────────── Sổ bài chưa nhận diện ─────────────────────────── */
/* Tab "Thống kê & Tra cứu". Sổ chỉ để tra cứu: không có gì ở đây đổi mức rủi ro,
 * và ghi chú của người thẩm định không phải dữ liệu quyền.
 *
 * URL phản ánh trạng thái (#thong-ke, #thong-ke/<unknown_id>): nút Back quay đúng tab,
 * và link tới một mục gửi được cho người thẩm định khác. */

const KIND_TEXT = { NOT_IDENTIFIED: 'Không nhận diện được', RIGHTS_UNKNOWN: 'Quyền chưa xác định' };
// Tầng nào của sổ nhận ra bài đã gặp (evidence.unknown_registry.match_method)
const METHOD_TEXT = {
  FINGERPRINT: 'dấu vân tay Chromaprint — gần như cùng bản ghi với lần gửi đầu',
  MERT: 'MERT — rất giống một đoạn sổ đã học của bài này (bản nén, cắt, chỉnh âm…)',
  COVER: 'Cover — cùng giai điệu nhưng đổi tông hoặc đổi nhịp',
};
const METHOD_SHORT = { FINGERPRINT: 'Vân tay', MERT: 'MERT', COVER: 'Cover' };
const STATUS_TEXT = { PENDING: 'Chờ thẩm định', REVIEWED: 'Đã thẩm định' };
const PLATFORM_TEXT = {
  YOUTUBE: 'YouTube', FACEBOOK: 'Facebook', TIKTOK: 'TikTok', OTHER: 'Khác', '—': 'Không rõ',
};
const RISK_LEVELS = ['LOW', 'CONDITIONAL', 'HIGH', 'UNKNOWN'];
const REGISTRY_PAGE = 20;
const REGISTRY_HASH = '#thong-ke';
const REGISTRY_ROUTE = /^#thong-ke(?:\/([0-9a-fA-F-]{36}))?$/;
// listSeq: phản hồi của lần tìm cũ về muộn không được đè kết quả của lần tìm mới
const registry = { offset: 0, total: 0, selectedId: null, listSeq: 0 };

/** Khớp unknown_registry_service.classify_unknown ở backend. */
function isUnknownResult(result) {
  const matchType = (result.match || {}).type;
  const risk = (result.assessment || {}).risk;
  return matchType === 'UNKNOWN' || (risk === 'UNKNOWN' && Boolean((result.identity || {}).recording_id));
}

const fmtNumber = (n) => Number(n || 0).toLocaleString('vi-VN');

/** Thời điểm ISO có múi giờ từ backend -> dd/mm/yyyy hh:mm theo giờ người xem. */
function fmtDateTime(value) {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString('vi-VN', {
    day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

/** 'YYYY-MM-DD' -> 'dd/mm'. Không qua Date: chuỗi chỉ có ngày bị hiểu là giờ UTC. */
function fmtDay(value, withYear = false) {
  const [y, m, d] = String(value).split('-');
  return withYear ? `${d}/${m}/${y}` : `${d}/${m}`;
}

/** Mức rủi ro dạng chữ (tooltip, câu thuần văn bản). */
function riskText(risk) {
  if (!risk) return '—';
  return RISK_TEXT[risk] ? `${risk} — ${RISK_TEXT[risk]}` : risk;
}

/** Mức rủi ro dạng HTML: chấm màu theo token trạng thái + mã chữ. Không dùng emoji —
 *  mỗi font vẽ ⚪🟢 một kiểu (Segoe UI Emoji vẽ ⚪ thành quả cầu tím), còn chấm CSS
 *  theo đúng token sáng/tối. Màu không bao giờ đứng một mình: luôn có chữ đi kèm. */
function riskHtml(risk) {
  if (!risk) return '—';
  const known = RISK_LEVELS.includes(risk) ? risk : 'UNKNOWN';
  return `<span class="risk-dot r-${known}" aria-hidden="true"></span>${escapeHtml(risk)}`;
}

/** Cập nhật URL theo mục đang mở mà KHÔNG thêm lịch sử (chọn / đóng một mục). */
function replaceHash(hash) {
  if (location.hash !== hash) history.replaceState(null, '', hash);
}

/** aria-busy + làm mờ nội dung CŨ trong lúc tải — không xoá trắng rồi vẽ lại (giật khung). */
function setBusy(el, busy) {
  el.setAttribute('aria-busy', busy ? 'true' : 'false');
  el.classList.toggle('is-loading', busy);
}

/** Thẻ sổ ở màn Kết quả — chỉ hiện khi kết quả là UNKNOWN. */
function renderRegistryCard(result) {
  const info = (result.evidence || {}).unknown_registry;
  const card = $('#card-registry');
  const button = $('#open-in-registry');
  const review = $('#registry-review');
  card.hidden = !info;
  review.hidden = true;
  if (!info) return;

  if (info.error_code) {
    $('#registry-summary').textContent = 'Kết quả UNKNOWN nhưng lần này không ghi được '
      + `vào sổ (${info.error_code}). Mức rủi ro ở trên không bị ảnh hưởng.`;
    button.hidden = true;
    return;
  }

  const kind = KIND_TEXT[info.kind] || info.kind;
  const label = [info.reviewer_title, info.reviewer_artist].filter(Boolean).join(' — ');
  const recognized = !info.is_new && info.kind === 'NOT_IDENTIFIED';
  $('#registry-heading').textContent = recognized ? 'Nhận ra bài đã gặp'
    : (info.is_new ? 'Bài mới — đã ghi vào sổ' : 'Sổ bài chưa nhận diện');

  let summary;
  if (info.is_new) {
    summary = `Lần đầu hệ thống gặp bài này — đã tạo mục mới trong sổ (${kind}).`;
  } else {
    const name = label ? `“${label}” (tên do người thẩm định đặt)`
      : (info.first_filename ? `bài lần đầu gửi với tên file “${info.first_filename}”` : 'một bài chưa đặt tên');
    summary = recognized
      ? `Đây là ${name}. Hệ thống đã gặp bài này ${fmtNumber(info.sighting_count)} lần, lần đầu `
        + `${fmtDateTime(info.first_seen_at)}.`
      : `Hệ thống đã gặp bài này ${fmtNumber(info.sighting_count)} lần, lần đầu `
        + `${fmtDateTime(info.first_seen_at)} — lượt này được gộp vào mục cũ (${kind}).`;
    if (info.match_method) {
      const fmt3 = (v) => v.toLocaleString('vi-VN', { minimumFractionDigits: 3, maximumFractionDigits: 3 });
      const score = info.match_score == null ? '—' : fmt3(info.match_score);
      const threshold = info.threshold == null ? '' : ` ≥ ngưỡng ${fmt3(info.threshold)}`;
      summary += ` Nhận ra qua ${METHOD_TEXT[info.match_method] || info.match_method}: điểm ${score}${threshold}`
        + `${info.oti_text ? ` (${info.oti_text})` : ''}.`;
    }
  }
  if (info.kind === 'NOT_IDENTIFIED' && info.fingerprint_available === false) {
    summary += ' Không sinh được fingerprint nên chưa đối chiếu được bằng dấu vân tay.';
  }
  $('#registry-summary').textContent = summary;

  // Sổ "học" bài này tới đâu: đặc trưng lưu lượt này + tổng của mục
  const learned = $('#registry-learned');
  let learnedText = '';
  if (info.features_error) {
    learnedText = 'Lần này không trích được đặc trưng để học — chỉ đối chiếu bằng dấu vân tay.';
  } else if (info.learning && info.features_learned > 0) {
    learnedText = `Đã ghi nhớ ${fmtNumber(info.features_learned)} đặc trưng từ lượt này `
      + `(tổng ${fmtNumber(info.features_total)}: vector MERT từng đoạn 15 giây + Cover) để nhận ra `
      + 'bài này ở lần sau, kể cả bản đã nén, cắt hoặc đổi tông. Không lưu file âm thanh.';
  } else if (info.learning && info.noise_like_segments > 0 && !info.features_total) {
    learnedText = 'Âm thanh này giống tiếng ồn nên không học bằng MERT (MERT coi mọi tiếng ồn '
      + 'gần như giống nhau) — chỉ nhận lại được khi gửi đúng bản ghi này.';
  } else if (info.learning && info.features_total > 0) {
    learnedText = `Sổ đã có ${fmtNumber(info.features_total)} đặc trưng của bài này; lượt này không có đoạn nào mới nên không lưu thêm.`;
  }
  learned.textContent = learnedText;
  learned.hidden = !learnedText;

  if (label || info.reviewer_note) {
    review.textContent = `Người thẩm định ghi (${STATUS_TEXT[info.review_status] || info.review_status}): `
      + [label, info.reviewer_note].filter(Boolean).join('. ');
    review.hidden = false;
  }
  button.hidden = false;
  button.dataset.id = info.unknown_id;
}

/* ── Tooltip dùng chung cho các biểu đồ (hover + focus bàn phím) ── */
function placeTooltip(x, y) {
  const tip = $('#chart-tooltip');
  const width = tip.offsetWidth;
  const left = Math.min(Math.max(8, x - width / 2), window.innerWidth - width - 8);
  tip.style.left = `${left}px`;
  tip.style.top = `${Math.max(8, y - tip.offsetHeight - 10)}px`;
}

function bindTooltip(container) {
  const tip = $('#chart-tooltip');
  const target = (event) => {
    const el = event.target.closest('[data-tip]');
    return el && container.contains(el) ? el : null;
  };
  container.addEventListener('pointermove', (event) => {
    const el = target(event);
    if (!el) { tip.hidden = true; return; }
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    placeTooltip(event.clientX, event.clientY);
  });
  container.addEventListener('pointerleave', () => { tip.hidden = true; });
  container.addEventListener('focusin', (event) => {
    const el = target(event);
    if (!el) return;
    const rect = el.getBoundingClientRect();
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    placeTooltip(rect.left + rect.width / 2, rect.top);
  });
  container.addEventListener('focusout', () => { tip.hidden = true; });
}

/* ── Biểu đồ ── */

/** Thanh ngang một chuỗi. Nhãn chữ + con số luôn nằm cạnh thanh, nên không có hàng
 *  nào phải đọc bằng màu. `labelHtml` (đã escape) thay cho `label` khi cần chấm màu. */
function renderHBars(container, rows) {
  const total = rows.reduce((sum, r) => sum + r.value, 0);
  const max = Math.max(0, ...rows.map((r) => r.value));
  if (!rows.length || max === 0) {
    container.innerHTML = '<p class="chart-empty">Chưa có dữ liệu.</p>';
    return;
  }
  container.innerHTML = rows.map((r) => {
    const share = total ? ` (${((r.value / total) * 100).toFixed(0)}%)` : '';
    const width = ((r.value / max) * 100).toFixed(1);
    const tip = escapeHtml(`${r.label}: ${fmtNumber(r.value)}${share}`);
    return `<div class="hbar-row" data-tip="${tip}">
      <span class="hbar-label">${r.labelHtml || escapeHtml(r.label)}</span>
      <span class="hbar-track" aria-hidden="true"><i class="hbar-fill ${r.fillClass || ''}"
        style="width:${width}%${r.value ? '' : ';min-width:0'}"></i></span>
      <span class="hbar-value">${fmtNumber(r.value)}</span></div>`;
  }).join('');
}

/** Mức trần "đẹp" cho trục: 1, 2, 5 × 10^k. */
function niceMax(value) {
  if (value <= 0) return 0;
  const power = 10 ** Math.floor(Math.log10(value));
  return [1, 2, 5, 10].map((m) => m * power).find((v) => v >= value);
}

/** Cột theo ngày, một chuỗi. Chỉ ghi số trên cột cao nhất; số của mọi ngày nằm ở
 *  tooltip (hover/Tab) và ở bảng "Xem dạng bảng" ngay dưới. */
function renderDays(container, days) {
  const max = Math.max(0, ...days.map((d) => d.count));
  if (!days.length || max === 0) {
    container.innerHTML = '<p class="chart-empty">Chưa có lượt gặp nào trong 30 ngày qua.</p>';
  } else {
    const top = niceMax(max);
    const peak = days.map((d) => d.count).lastIndexOf(max);
    const cols = days.map((d, i) => {
      const height = (d.count / top) * 100;
      const label = `${fmtDay(d.date, true)}: ${fmtNumber(d.count)} lượt gặp`;
      const cap = i === peak
        ? `<span class="vcol-cap" style="bottom:calc(${height.toFixed(1)}% + 2px)">${fmtNumber(d.count)}</span>` : '';
      return `<div class="vcol" tabindex="0" role="img" aria-label="${escapeHtml(label)}"
        data-tip="${escapeHtml(label)}"><i style="height:${height.toFixed(1)}%${d.count ? ';min-height:2px' : ''}"></i>${cap}</div>`;
    }).join('');
    const mid = days[Math.floor(days.length / 2)];
    container.innerHTML = `
      <div class="vchart-y" aria-hidden="true"><span>${fmtNumber(top)}</span><span>0</span></div>
      <div class="vchart-plot">${cols}</div>
      <div class="vchart-x" aria-hidden="true"><span>${fmtDay(days[0].date)}</span>
        <span>${fmtDay(mid.date)}</span><span>${fmtDay(days[days.length - 1].date)}</span></div>`;
  }

  $('#table-days tbody').innerHTML = days.slice().reverse().map((d) =>
    `<tr><td>${fmtDay(d.date, true)}</td><td class="num">${fmtNumber(d.count)}</td></tr>`).join('');
}

function statTile(label, value, sub) {
  return `<div class="stat"><p class="stat-label">${escapeHtml(label)}</p>
    <p class="stat-value">${escapeHtml(value)}</p>
    ${sub ? `<p class="stat-sub">${escapeHtml(sub)}</p>` : ''}</div>`;
}

function renderStats(stats) {
  const totals = stats.totals || {};
  const jobs = stats.jobs || {};
  const rate = jobs.done_since_start
    ? `${((jobs.registered / jobs.done_since_start) * 100).toFixed(0)}%` : '—';
  $('#registry-disabled').hidden = stats.registry_enabled !== false;
  $('#stat-grid').innerHTML = [
    statTile('Mục trong sổ', fmtNumber(totals.entries)),
    statTile('Chờ thẩm định', fmtNumber(totals.pending)),
    statTile('Đã thẩm định', fmtNumber(totals.reviewed)),
    statTile('Tổng lượt gặp', fmtNumber(totals.sightings)),
    statTile('Mục gặp lại ≥ 2 lần', fmtNumber(totals.resighted_entries)),
    statTile('Job bị ghi sổ', rate,
      jobs.done_since_start
        ? `${fmtNumber(jobs.registered)} / ${fmtNumber(jobs.done_since_start)} job từ khi có sổ` : ''),
  ].join('');

  const byKind = stats.by_kind || {};
  renderHBars($('#chart-kind'), Object.keys(KIND_TEXT).map((k) => ({
    label: KIND_TEXT[k], value: byKind[k] || 0,
  })));

  const byRisk = stats.by_last_risk || {};
  renderHBars($('#chart-risk'), RISK_LEVELS
    .filter((r) => byRisk[r])
    .map((r) => ({ label: riskText(r), labelHtml: riskHtml(r), value: byRisk[r], fillClass: `fill-${r}` })));

  const byPlatform = stats.by_platform || {};
  renderHBars($('#chart-platform'), Object.entries(byPlatform)
    .sort((a, b) => b[1] - a[1])
    .map(([p, n]) => ({ label: PLATFORM_TEXT[p] || p, value: n })));

  renderDays($('#chart-days'), stats.sightings_by_day || []);

  const top = stats.top_resighted || [];
  $('#top-resighted').innerHTML = top.length
    ? top.map((item) => `<li><button type="button" class="linklike" data-id="${escapeHtml(item.unknown_id)}">`
      + `${escapeHtml(item.label)}</button> <span class="count">· ${fmtNumber(item.sighting_count)} lượt</span></li>`).join('')
    : '<li class="chart-empty">Chưa có bài nào được gửi lại.</li>';
}

/* ── Danh sách & chi tiết ── */

function hasFilters() {
  return Boolean($('#rs-q').value.trim() || $('#rs-kind').value || $('#rs-status').value);
}

function registryQuery() {
  const params = new URLSearchParams({
    sort: $('#rs-sort').value, limit: REGISTRY_PAGE, offset: registry.offset,
  });
  const q = $('#rs-q').value.trim();
  if (q) params.set('q', q);
  if ($('#rs-kind').value) params.set('kind', $('#rs-kind').value);
  if ($('#rs-status').value) params.set('status', $('#rs-status').value);
  return params;
}

/** Trạng thái rỗng phải nói bước tiếp theo, không chỉ "không có gì". */
function emptyListRow() {
  const body = hasFilters()
    ? 'Không có mục nào khớp từ khoá / bộ lọc. '
      + '<button type="button" class="linklike" data-action="clear-filters">Xoá bộ lọc</button>'
    : 'Sổ đang trống. Mỗi kết quả UNKNOWN ở tab <a href="#phan-tich">Phân tích</a> '
      + 'sẽ tự xuất hiện ở đây.';
  return `<tr><td colspan="6" class="empty-cell">${body}</td></tr>`;
}

function renderList(data) {
  registry.total = data.total || 0;
  const items = data.items || [];
  // Ô đầu là NÚT thật: trình đọc màn hình đọc được "nút", Tab/Enter/Space hoạt động
  // sẵn. Bấm vào chỗ khác trên hàng vẫn mở chi tiết (tiện cho chuột).
  $('#registry-table tbody').innerHTML = items.length
    ? items.map((item) => `<tr data-id="${escapeHtml(item.unknown_id)}"
        class="${item.unknown_id === registry.selectedId ? 'selected' : ''}">
        <td><button type="button" class="row-link" data-id="${escapeHtml(item.unknown_id)}"
          aria-label="Xem chi tiết: ${escapeHtml(item.label)}">${escapeHtml(item.label)}</button></td>
        <td>${escapeHtml(KIND_TEXT[item.kind] || item.kind)}</td>
        <td class="num">${fmtNumber(item.sighting_count)}</td>
        <td>${escapeHtml(fmtDateTime(item.last_seen_at))}</td>
        <td>${riskHtml(item.last_risk_level)}</td>
        <td>${escapeHtml(STATUS_TEXT[item.review_status] || item.review_status)}</td></tr>`).join('')
    : emptyListRow();

  const from = registry.total ? registry.offset + 1 : 0;
  const to = Math.min(registry.offset + items.length, registry.total);
  // role=status: trình đọc màn hình báo số kết quả sau mỗi lần tìm/lọc/chuyển trang
  $('#rs-range').textContent = registry.total
    ? `Hiển thị ${fmtNumber(from)}–${fmtNumber(to)} trên ${fmtNumber(registry.total)} mục`
    : 'Không có mục nào';
  $('#rs-prev').disabled = registry.offset <= 0;
  $('#rs-next').disabled = registry.offset + REGISTRY_PAGE >= registry.total;
}

function showRegistryError(err) {
  const box = $('#registry-error');
  if (!err) { box.hidden = true; return; }
  box.textContent = err.message || String(err);
  box.hidden = false;
}

async function loadStats() {
  const panel = $('#registry-stats');
  setBusy(panel, true);
  try {
    // Chia lượt gặp theo ngày ở giờ của người xem (getTimezoneOffset: UTC+7 -> -420)
    const offset = -new Date().getTimezoneOffset();
    renderStats(await fetchJson(`${API}/unknown-tracks/stats?utc_offset_min=${offset}`));
  } catch (err) {
    showRegistryError(err);
  } finally {
    setBusy(panel, false);
  }
}

async function loadList() {
  const panel = $('#registry-search-card');
  const seq = ++registry.listSeq;
  setBusy(panel, true);
  try {
    const data = await fetchJson(`${API}/unknown-tracks?${registryQuery()}`);
    if (seq === registry.listSeq) renderList(data);
  } catch (err) {
    if (seq === registry.listSeq) showRegistryError(err);
  } finally {
    if (seq === registry.listSeq) setBusy(panel, false);
  }
}

function renderDetail(entry) {
  registry.selectedId = entry.unknown_id;
  $$('#registry-table tbody tr').forEach((tr) =>
    tr.classList.toggle('selected', tr.dataset.id === entry.unknown_id));

  $('#rd-title').textContent = entry.label;
  const catalogue = entry.catalogue;
  const rows = [
    ['Loại', KIND_TEXT[entry.kind] || entry.kind],
    ['Lần đầu gặp', fmtDateTime(entry.first_seen_at)],
    ['Gặp gần nhất', fmtDateTime(entry.last_seen_at)],
    ['Số lượt gặp', fmtNumber(entry.sighting_count)],
    ['Loại khớp gần nhất', entry.last_match_type],
    ['Rủi ro gần nhất', null, {
      html: [riskHtml(entry.last_risk_level), escapeHtml(entry.last_category || '')]
        .filter(Boolean).join(' · '),
    }],
    ['Điều kiện gần nhất', entry.last_condition],
    // RIGHTS_UNKNOWN: bài có trong kho nhưng quyền chưa xác định — nêu rõ bản ghi nào
    ...(catalogue ? [
      ['Bản ghi trong kho', [catalogue.title, catalogue.artist].filter(Boolean).join(' — ')
        + (catalogue.source_dataset ? ` (${catalogue.source_dataset})` : '')],
      ['recording_id', entry.recording_id, { mono: true }],
    ] : []),
    ['Độ dài fingerprint (s)', entry.fingerprint_duration],
    // Sổ đã "học" bài này tới đâu (chỉ mục không nhận diện được mới có)
    ...(entry.learned && entry.kind === 'NOT_IDENTIFIED' ? [
      ['Đặc trưng đã học', `${fmtNumber(entry.learned.mert)} đoạn MERT · ${fmtNumber(entry.learned.cover)} Cover`],
    ] : []),
    ['Trạng thái', STATUS_TEXT[entry.review_status] || entry.review_status],
    ['Thẩm định lúc', entry.reviewed_at ? fmtDateTime(entry.reviewed_at) : null],
    ['unknown_id', entry.unknown_id, { mono: true }],
  ];
  // Chế độ EVIDENCE_DETAIL=public không trả tên file
  if (entry.first_filename) rows.splice(1, 0, ['Tên file lần đầu', entry.first_filename]);
  dl($('#rd-info'), rows);

  const hint = entry.hint;
  const hintBox = $('#rd-hint');
  hintBox.hidden = !hint;
  if (hint) {
    const name = [hint.title, hint.artist].filter(Boolean).join(' — ') || hint.recording_id;
    hintBox.textContent = `Gợi ý: ${name} (cosine MERT ${hint.score ?? '—'}). ${hint.note}`;
  }

  const sightings = entry.sightings || [];
  $('#rd-sightings tbody').innerHTML = sightings.length
    ? sightings.map((s) => `<tr>
        <td>${escapeHtml(fmtDateTime(s.seen_at))}</td>
        <td>${escapeHtml(PLATFORM_TEXT[s.platform] || show(s.platform))}</td>
        <td>${escapeHtml(show(s.commercial_use))}</td>
        <td>${escapeHtml(show(s.monetization))}</td>
        <td>${riskHtml(s.risk_level)}</td>
        <td>${escapeHtml(METHOD_SHORT[s.match_method] || (s.match_score == null ? 'Mục mới' : '—'))}</td>
        <td class="num">${escapeHtml(show(s.match_score))}</td></tr>`).join('')
    : '<tr><td colspan="7" class="muted">Chưa có lượt gặp.</td></tr>';

  $('#rv-title').value = entry.reviewer_title || '';
  $('#rv-artist').value = entry.reviewer_artist || '';
  $('#rv-note').value = entry.reviewer_note || '';
  $('#rv-status').value = entry.review_status || 'PENDING';
  $('#rv-msg').textContent = '';
  $('#registry-detail').hidden = false;
}

/** Mở chi tiết, cuộn tới và CHUYỂN FOCUS vào tiêu đề — không thì người dùng bàn phím
 *  phải Tab qua hết bảng và nút phân trang mới tới được khung vừa mở. */
async function openDetail(unknownId) {
  try {
    renderDetail(await fetchJson(`${API}/unknown-tracks/${encodeURIComponent(unknownId)}`));
    replaceHash(`${REGISTRY_HASH}/${unknownId}`);
    $('#registry-detail').scrollIntoView({ behavior: scrollBehavior(), block: 'start' });
    $('#rd-title').focus({ preventScroll: true });
  } catch (err) {
    showRegistryError(err);
  }
}

/** Đóng chi tiết; `returnFocus` trả focus về đúng nút đã mở nó. */
function closeDetail({ returnFocus = false } = {}) {
  const lastId = registry.selectedId;
  $('#registry-detail').hidden = true;
  registry.selectedId = null;
  $$('#registry-table tbody tr').forEach((tr) => tr.classList.remove('selected'));
  if (state.view === 'registry') replaceHash(REGISTRY_HASH);
  if (returnFocus) {
    const opener = lastId && $$('#registry-table .row-link').find((b) => b.dataset.id === lastId);
    (opener || $('#rs-q')).focus();
  }
}

async function saveReview(event) {
  event.preventDefault();
  if (!registry.selectedId) return;
  const form = $('#review-form');
  const button = $('#rv-save');
  const msg = $('#rv-msg');
  msg.textContent = 'Đang lưu…';
  button.disabled = true;               // chặn bấm hai lần gửi hai PATCH
  setBusy(form, true);
  try {
    const entry = await fetchJson(`${API}/unknown-tracks/${encodeURIComponent(registry.selectedId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      // Gửi đủ các trường: ô để trống nghĩa là xoá giá trị cũ
      body: JSON.stringify({
        title: $('#rv-title').value,
        artist: $('#rv-artist').value,
        note: $('#rv-note').value,
        review_status: $('#rv-status').value,
      }),
    });
    renderDetail(entry);
    $('#rv-msg').textContent = 'Đã lưu ghi chú.';
    loadList();
    loadStats();
  } catch (err) {
    msg.textContent = err.message;
  } finally {
    button.disabled = false;
    setBusy(form, false);
  }
}

/** Mở tab Thống kê & Tra cứu; `unknownId` thì mở luôn chi tiết mục đó. */
function openRegistry(unknownId) {
  showScreen('registry');
  showRegistryError(null);
  loadStats();
  loadList();
  if (unknownId) openDetail(unknownId);
  else if (registry.selectedId) closeDetail();
}

/** URL -> màn. Chạy khi tải trang (không có event) và mỗi lần hash đổi (bấm tab,
 *  Back/Forward, mở một link có sẵn). */
function route(event) {
  const registryMatch = location.hash.match(REGISTRY_ROUTE);
  if (registryMatch) {
    openRegistry(registryMatch[1]);
    return;
  }
  // Bấm tab: focus ở lại tab (như mọi thanh tab). Back/Forward, nút CTA, link trong trang:
  // focus vào heading của màn mới. Tải trang (không có event): không kéo focus.
  const viaTab = state.viaTab;
  state.viaTab = false;
  const moveFocus = Boolean(event) && !viaTab;
  const fromOtherTab = state.view !== 'analyze';
  const analyzeMatch = location.hash.match(ANALYZE_ROUTE);
  if (!analyzeMatch) {
    // #lich-su, #trang-chu; hash rỗng hoặc lạ -> Trang chủ
    const view = location.hash === HISTORY_HASH ? 'history' : 'home';
    replaceHash(view === 'history' ? HISTORY_HASH : HOME_HASH);
    if (view === 'history') renderHistory();
    showScreen(view);
    if (moveFocus) focusScreenHeading(view);
    return;
  }
  let target = ANALYZE_SUBROUTE[analyzeMatch[1]];
  // Tải lại trang / link cũ tới #phan-tich/ket-qua: kết quả không còn trong bộ nhớ
  if (target && (!state.result || state.polling)) target = null;
  if (!target) target = fromOtherTab ? state.lastAnalyzeScreen : (state.polling ? 'processing' : 'upload');
  goAnalyze(target, { push: false, focus: moveFocus });
  // Màn vừa hiện: canvas sóng âm mới đo được bề rộng
  if (target === 'processing') drawWave($('#proc-wave'), wave.peaks || decorativePeaks('pipeline'), wave.progress);
}

function initRegistry() {
  // Tab là liên kết (#phan-tich / #thong-ke): Back/Forward, mở tab mới, sao chép link
  // đều đúng; mọi chuyển tab đi qua `route`.
  window.addEventListener('hashchange', route);

  $('#open-in-registry').addEventListener('click', (event) => {
    const hash = `${REGISTRY_HASH}/${event.currentTarget.dataset.id}`;
    if (location.hash === hash) route();
    else location.hash = hash;
  });

  $('#registry-search').addEventListener('submit', (event) => {
    event.preventDefault();
    registry.offset = 0;
    loadList();
  });
  ['#rs-kind', '#rs-status', '#rs-sort'].forEach((sel) => $(sel).addEventListener('change', () => {
    registry.offset = 0;
    loadList();
  }));
  $('#rs-prev').addEventListener('click', () => {
    registry.offset = Math.max(0, registry.offset - REGISTRY_PAGE);
    loadList();
  });
  $('#rs-next').addEventListener('click', () => {
    registry.offset += REGISTRY_PAGE;
    loadList();
  });

  $('#registry-table tbody').addEventListener('click', (event) => {
    if (event.target.closest('[data-action="clear-filters"]')) {
      $('#rs-q').value = '';
      $('#rs-kind').value = '';
      $('#rs-status').value = '';
      registry.offset = 0;
      loadList();
      $('#rs-q').focus();
      return;
    }
    const row = event.target.closest('tr[data-id]');
    if (row) openDetail(row.dataset.id);
  });
  $('#top-resighted').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-id]');
    if (button) openDetail(button.dataset.id);
  });

  $('#rd-close').addEventListener('click', () => closeDetail({ returnFocus: true }));
  $('#review-form').addEventListener('submit', saveReview);

  ['#chart-kind', '#chart-risk', '#chart-platform', '#chart-days'].forEach((sel) => bindTooltip($(sel)));

  route();   // mở đúng tab nếu trang được mở bằng link #thong-ke/...
}

/* ─────────────────────────── Âm nhạc: sóng âm, số đếm, hiện dần ─────────────────────────── */

// File audio nhỏ hơn mức này mới giải mã để vẽ sóng âm (giải mã = PCM trong RAM, ~10× dung
// lượng MP3). Video hoặc file lớn hơn: sóng âm trang trí, không giả là của file.
const WAVE_DECODE_MAX_BYTES = 15 * 1024 * 1024;
const WAVE_BARS = 96;
const AUDIO_EXTENSIONS = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus'];
const wave = { peaks: null, decoded: false, seq: 0, progress: 0 };

/** Màu sóng âm lấy từ token CSS (--color-wave-*), để canvas đổi theo chế độ sáng/tối và
 *  giữ tương phản ≥ 3:1 giữa phần đã tô và phần chưa tới (docs/DESIGN_SYSTEM.md §2). */
function waveColors() {
  const css = getComputedStyle(document.documentElement);
  const read = (name) => css.getPropertyValue(name).trim();
  return {
    stops: [read('--color-wave-from'), read('--color-wave-to')],
    empty: read('--color-wave-empty'),
  };
}

/** Vẽ sóng âm dạng cột; `progress` 0..1 = phần đã tô gradient (phần còn lại màu nhạt). */
function drawWave(canvas, peaks, progress = 1) {
  if (!canvas || !peaks || !peaks.length) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;   // màn đang ẩn: vẽ lại khi hiện (resize / chuyển màn)
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const { stops, empty } = waveColors();
  const gradient = ctx.createLinearGradient(0, 0, rect.width, 0);
  stops.forEach((color, i) => gradient.addColorStop(i / (stops.length - 1), color));
  const step = rect.width / peaks.length;
  const barWidth = Math.max(1.5, step * 0.62);
  const mid = rect.height / 2;
  peaks.forEach((peak, i) => {
    const h = Math.max(2, peak * (rect.height - 2));
    const x = i * step + (step - barWidth) / 2;
    ctx.fillStyle = (i + 0.5) / peaks.length <= progress ? gradient : empty;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, mid - h / 2, barWidth, h, barWidth / 2);
    else ctx.rect(x, mid - h / 2, barWidth, h);
    ctx.fill();
  });
}

/** Sóng trang trí khi không giải mã được file (video, file lớn): hình dạng suy từ tên file,
 *  cố định cho cùng một file — KHÔNG phải biên độ thật. */
function decorativePeaks(seed) {
  let h = 2166136261;
  for (const ch of String(seed)) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return Array.from({ length: WAVE_BARS }, (_, i) => {
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    const noise = ((h >>> 0) % 1000) / 1000;
    const envelope = 0.55 + 0.45 * Math.sin((i / WAVE_BARS) * Math.PI);
    return 0.18 + 0.72 * envelope * (0.35 + 0.65 * noise);
  });
}

/** Giải mã file audio ngay trong trình duyệt (không gửi đi đâu) và rút ra WAVE_BARS đỉnh. */
async function decodePeaks(file) {
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Ctx) return null;
  const buffer = await new Ctx(1, 1, 44100).decodeAudioData(await file.arrayBuffer());
  const data = buffer.getChannelData(0);
  const size = Math.floor(data.length / WAVE_BARS) || 1;
  const peaks = [];
  for (let b = 0; b < WAVE_BARS; b += 1) {
    let max = 0;
    const end = Math.min(data.length, (b + 1) * size);
    for (let i = b * size; i < end; i += 16) max = Math.max(max, Math.abs(data[i]));
    peaks.push(max);
  }
  const top = Math.max(...peaks) || 1;
  return peaks.map((v) => v / top);
}

/** File vừa chọn: sóng âm thật nếu giải mã được, nếu không thì sóng trang trí cho màn Xử lý. */
async function prepareWave(file) {
  const seq = ++wave.seq;
  const canvas = $('#file-wave');
  wave.peaks = file ? decorativePeaks(file.name) : null;
  wave.decoded = false;
  canvas.hidden = true;
  if (!file) return;
  if (!AUDIO_EXTENSIONS.includes(fileExt(file.name)) || file.size > WAVE_DECODE_MAX_BYTES) return;
  try {
    const peaks = await decodePeaks(file);
    if (seq !== wave.seq || !peaks) return;       // người dùng đã chọn file khác
    wave.peaks = peaks;
    wave.decoded = true;
    canvas.hidden = false;
    drawWave(canvas, peaks, 1);
  } catch (_) {
    // Trình duyệt không giải mã được định dạng này: giữ sóng trang trí, không báo lỗi —
    // máy chủ mới là nơi quyết định file có hợp lệ hay không
  }
}

/** Thanh tiến trình sóng âm của màn Xử lý: tô tới bước pipeline THẬT (backend báo). */
function paintProgress(done, total = STAGE_ORDER.length, text = '') {
  wave.progress = total ? done / total : 0;
  const bar = $('#proc-progress');
  bar.setAttribute('aria-valuenow', String(done));
  bar.setAttribute('aria-valuetext', text || `${done}/${total} bước`);
  drawWave($('#proc-wave'), wave.peaks || decorativePeaks('pipeline'), wave.progress);
}

/** Số đếm chạy lên tới `to` (ease-out). Giảm chuyển động: hiện ngay con số cuối. */
function countUp(el, to, { decimals = 1, duration = 900 } = {}) {
  const format = (v) => v.toLocaleString('vi-VN', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  if (el.countFrame) cancelAnimationFrame(el.countFrame);
  if (reducedMotion() || !Number.isFinite(to)) {
    el.textContent = Number.isFinite(to) ? format(to) : '—';
    return;
  }
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    el.textContent = format(to * (1 - (1 - t) ** 3));
    if (t < 1) el.countFrame = requestAnimationFrame(tick);
  };
  el.countFrame = requestAnimationFrame(tick);
}

/** Thẻ kết quả hiện lần lượt: đánh số thứ tự cho các thẻ đang hiện (CSS dùng --i làm độ trễ). */
function staggerReveal() {
  $$('#screen-result .reveal').filter((el) => !el.hidden)
    .forEach((el, i) => el.style.setProperty('--i', String(i)));
}

/* ─────────────────────────── Tablist ─────────────────────────── */

/** Tablist theo mẫu ARIA: Trái/Phải/Home/End chuyển thẻ, chỉ thẻ đang chọn nằm trong thứ
 *  tự Tab. Dùng cho thẻ thể loại ví dụ ở màn Kết quả. */
function initTablist(tablist, onSelect) {
  const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
  const select = (tab, focus = false) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      const panel = t.getAttribute('aria-controls');
      if (panel && document.getElementById(panel)) document.getElementById(panel).hidden = !on;
    });
    if (focus) tab.focus();
    if (onSelect) onSelect(tab);
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', (event) => {
      const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      select(tabs[(next + tabs.length) % tabs.length], true);
    });
  });
  return select;
}

/* ─────────────────────────── Thể loại (VÍ DỤ minh hoạ) ─────────────────────────── */
/* Hệ thống CHƯA phân loại thể loại. Dữ liệu dưới đây cố định, giống nhau cho mọi file, chỉ
 * để minh hoạ giao diện — thẻ có viền nét đứt, nhãn "Ví dụ minh hoạ" và thanh gạch chéo. */
const SAMPLE_GENRES = [
  { id: 'pop', name: 'Pop', value: 0.62, text: 'Nhạc phổ thông, giai điệu dễ nhớ, cấu trúc verse – chorus rõ ràng.' },
  { id: 'bolero', name: 'Bolero', value: 0.18, text: 'Nhịp chậm 4/4 đặc trưng, lời tự sự, phổ biến ở nhạc vàng Việt Nam.' },
  { id: 'rap', name: 'Rap / Hip-hop', value: 0.09, text: 'Lời nói theo nhịp trên nền beat; thường có sample từ bản ghi khác.' },
  { id: 'edm', name: 'EDM', value: 0.07, text: 'Nhạc điện tử cho sàn nhảy: drop, build-up, tempo 120–130 BPM.' },
  { id: 'dan-ca', name: 'Dân ca', value: 0.04, text: 'Làn điệu truyền thống các vùng miền; tác phẩm gốc nhiều khi đã thuộc phạm vi công cộng.' },
];

function initGenreSample() {
  const list = $('.genre-tabs');
  const panel = $('#genre-panel');
  list.innerHTML = SAMPLE_GENRES.map((g, i) => `<button type="button" role="tab" class="chip"
      id="genre-tab-${g.id}" aria-controls="genre-panel" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}">${g.name}</button>`).join('');
  initTablist(list, (tab) => {
    const genre = SAMPLE_GENRES.find((g) => `genre-tab-${g.id}` === tab.id);
    panel.setAttribute('aria-labelledby', tab.id);
    // Gỡ rồi gắn lại để animation "trượt vào" chạy mỗi lần đổi thẻ
    panel.style.animation = 'none';
    void panel.offsetWidth;
    panel.style.animation = '';
    panel.innerHTML = `<h4>${genre.name} — ${Math.round(genre.value * 100)}% (ví dụ)</h4>
      <div class="sample-bar"><i style="width:${genre.value * 100}%"></i></div>
      <p>${genre.text}</p>`;
    panel.hidden = false;
  })(list.querySelector('[role="tab"]'));
}

/* ─────────────────────────── Lịch sử (lưu trong trình duyệt này) ─────────────────────────── */
/* Chỉ các job DO TRÌNH DUYỆT NÀY chạy: job_id là khoá truy cập (docs/SECURITY.md §1), không
 * có API liệt kê job của người khác. Lưu localStorage — riêng máy này, xoá được bất cứ lúc nào. */
const HISTORY_KEY = 'mra.history.v1';
const HISTORY_MAX = 50;
const history_ = { filter: '' };

function readHistory() {
  try {
    const items = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    return Array.isArray(items) ? items : [];
  } catch (_) {
    return [];   // trình duyệt chặn lưu trữ / dữ liệu hỏng: coi như chưa có lịch sử
  }
}

function writeHistory(items) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX)));
  } catch (_) { /* chế độ riêng tư chặn lưu: bỏ qua, app vẫn chạy */ }
}

function recordHistory(jobId, result, fileName) {
  const { identity = {}, match = {}, assessment = {} } = result;
  const items = readHistory().filter((item) => item.jobId !== jobId);
  items.unshift({
    jobId,
    file: fileName || '—',
    at: new Date().toISOString(),
    risk: assessment.risk || 'UNKNOWN',
    matchType: match.type || null,
    confidence: match.confidence ?? null,
    track: identity.track || null,
    artist: identity.artist || null,
  });
  writeHistory(items);
}

function renderHistory() {
  const all = readHistory();
  const items = history_.filter ? all.filter((item) => item.risk === history_.filter) : all;
  const list = $('#history-list');
  $('#history-empty').hidden = all.length > 0;
  $('#history-clear').hidden = all.length === 0;
  $('.history-toolbar').hidden = all.length === 0;
  $('#history-status').textContent = all.length
    ? `${items.length} / ${all.length} lần kiểm tra${history_.filter ? ` · lọc: ${riskText(history_.filter)}` : ''}` : '';
  list.innerHTML = items.map((item) => {
    const who = [item.track, item.artist].filter(Boolean).map(escapeHtml).join(' — ') || 'Chưa định danh được bài';
    const conf = item.confidence == null ? '' : ` · tin cậy ${(item.confidence * 100).toLocaleString('vi-VN', { maximumFractionDigits: 1 })}%`;
    return `<li class="history-item risk-${escapeHtml(item.risk)}" data-job="${escapeHtml(item.jobId)}">
      <button type="button" class="history-open" data-job="${escapeHtml(item.jobId)}">
        <span class="history-file">${escapeHtml(item.file)}</span>
        <span class="history-risk">${riskHtml(item.risk)}</span>
        <span class="history-meta">${who} · ${escapeHtml(item.matchType || '—')}${conf} · ${fmtDateTime(item.at)}</span>
      </button>
      <button type="button" class="history-delete" data-job="${escapeHtml(item.jobId)}" aria-label="Xoá ${escapeHtml(item.file)} khỏi lịch sử">✕</button>
    </li>`;
  }).join('') || (all.length ? '<li class="muted">Không có mục nào ở mức rủi ro này.</li>' : '');
}

/** Mở lại một kết quả cũ. Máy chủ có thể không còn job (CSDL làm lại, job hết hạn): báo ngay
 *  tại mục đó, không đá người dùng sang màn khác. */
async function openHistoryItem(jobId, button) {
  const item = button.closest('.history-item');
  item.querySelector('.history-error')?.remove();
  setBusy(item, true);
  try {
    const result = await fetchJson(`${API}/results/${jobId}`);
    stopPolling();
    state.jobId = jobId;
    state.result = result;
    markAllStagesDone(result);
    renderResult(result);
    renderEvidence(result);
    goAnalyze('result');
  } catch (err) {
    const note = document.createElement('span');
    note.className = 'history-error';
    note.textContent = `Không mở lại được: ${err.message}`;
    button.append(note);
  } finally {
    setBusy(item, false);
  }
}

function initHistory() {
  $$('.filter-chips .chip').forEach((chip) => chip.addEventListener('click', () => {
    history_.filter = chip.dataset.risk;
    $$('.filter-chips .chip').forEach((c) => c.setAttribute('aria-pressed', String(c === chip)));
    renderHistory();
  }));
  $('#history-list').addEventListener('click', (event) => {
    const del = event.target.closest('.history-delete');
    if (del) {
      writeHistory(readHistory().filter((item) => item.jobId !== del.dataset.job));
      renderHistory();
      $('#history-status').textContent = 'Đã xoá một mục khỏi lịch sử.';
      ($('.history-open') || $('#screen-history h2')).focus();
      return;
    }
    const open = event.target.closest('.history-open');
    if (open) openHistoryItem(open.dataset.job, open);
  });
  $('#history-clear').addEventListener('click', () => {
    writeHistory([]);
    renderHistory();
    $('#screen-history h2').focus();
  });
}

/* ─────────────────────────── Phản hồi ─────────────────────────── */

async function sendFeedback(verdict, button) {
  const msg = $('#feedback-msg');
  const buttons = $$('.feedback-buttons button');
  buttons.forEach((b) => { b.disabled = true; });   // bấm hai lần = gửi hai phản hồi
  try {
    const response = await fetch(`${API}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        job_id: state.jobId,
        recording_id: (state.result.identity || {}).recording_id || null,
        verdict,
      }),
    });
    if (!response.ok) throw new Error(await apiError(response));
    const data = await response.json();
    $$('.feedback-buttons button').forEach((b) => b.setAttribute('aria-pressed', 'false'));
    button.setAttribute('aria-pressed', 'true');
    msg.textContent = data.message;
  } catch (err) {
    msg.textContent = err.message;
  } finally {
    buttons.forEach((b) => { b.disabled = false; });
  }
  msg.hidden = false;
}

/* ─────────────────────────── Khởi động ─────────────────────────── */

/** "Phân tích file khác": bỏ file đang chọn, về màn Tải lên. Kết quả cũ vẫn giữ trong bộ
 *  nhớ để nút Back / bước 3 xem lại được; lần gửi file mới sẽ thay nó. */
function restart() {
  stopPolling();
  $('#file').value = '';
  pickFile(null);
  goAnalyze('upload');
}

document.addEventListener('DOMContentLoaded', () => {
  initUpload();
  initGenreSample();
  initHistory();
  initRegistry();
  checkHealth();

  // Bấm tab: đánh dấu để route() không kéo focus khỏi tab (chỉ khi hash thật sự đổi)
  $$('.tabs .tab').forEach((tab) => tab.addEventListener('click', () => {
    state.viaTab = location.hash !== tab.getAttribute('href');
  }));

  // Canvas vẽ theo bề rộng thật: đổi cỡ cửa sổ thì vẽ lại (gộp theo khung hình)
  let resizeFrame = 0;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      if (wave.decoded) drawWave($('#file-wave'), wave.peaks, 1);
      drawWave($('#proc-wave'), wave.peaks || decorativePeaks('pipeline'), wave.progress);
    });
  });

  $('#to-evidence').addEventListener('click', () => goAnalyze('evidence'));
  $('#back-to-result').addEventListener('click', () => goAnalyze('result'));
  $('#restart').addEventListener('click', restart);
  $('#restart2').addEventListener('click', restart);
  $('#cancel-poll').addEventListener('click', () => { stopPolling(); goAnalyze('upload', { push: false }); });

  $$('.feedback-buttons button').forEach((button) =>
    button.addEventListener('click', () => sendFeedback(button.dataset.verdict, button)));

  // Thanh bước: nút bị tắt khi chưa tới được (updateSteps). Bước 1 KHÔNG xoá kết quả
  // như trước — chỉ quay về màn Tải lên, file và kết quả cũ vẫn còn.
  $$('.steps .step').forEach((button) => button.addEventListener('click', () => {
    if (!button.disabled) goAnalyze(button.dataset.step);
  }));
  updateSteps('upload');

  // Bấm lại tab Phân tích khi đang ở nó: không làm gì (link #phan-tich sẽ đưa về Tải lên)
  $('.tab[data-view="analyze"]').addEventListener('click', (event) => {
    if (state.view === 'analyze') {
      event.preventDefault();
      state.viaTab = false;   // không có điều hướng nào: đừng để cờ sót sang lần Back sau
    }
  });

  // Link bỏ qua: focus vào heading của màn đang mở, không đổi hash (#main không phải một màn)
  $('.skip-link').addEventListener('click', (event) => {
    event.preventDefault();
    ($('.screen.active h2') || $('#main')).focus();
  });

  // Khối trạng thái hệ thống: bấm ra ngoài hoặc Esc thì đóng, như một bảng nổi
  const health = $('#health');
  document.addEventListener('click', (event) => {
    if (health.open && !health.contains(event.target)) health.open = false;
  });
  health.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && health.open) {
      health.open = false;
      health.querySelector('summary').focus();
    }
  });
});
