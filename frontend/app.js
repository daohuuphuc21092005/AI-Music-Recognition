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

// Thứ tự phải khớp <li data-stage> trong index.html
const STAGE_ORDER = [
  'VALIDATING', 'EXTRACTING_AUDIO', 'FINGERPRINTING',
  'EMBEDDING', 'VECTOR_SEARCH', 'COVER_SEARCH', 'RIGHTS_LOOKUP', 'RULE_ENGINE',
];

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

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = { file: null, jobId: null, result: null, timer: null };

/* ─────────────────────────── Tiện ích ─────────────────────────── */

function showScreen(name) {
  $$('.screen').forEach((s) => s.classList.toggle('active', s.id === `screen-${name}`));
  const reached = ['upload', 'processing', 'result', 'evidence'].indexOf(name);
  $$('.steps li').forEach((li, i) => {
    li.classList.toggle('active', i === reached);
    li.classList.toggle('done', i < reached);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
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
    if (opts.mono) dd.className = 'mono';

    if (opts.html) {
      dd.innerHTML = opts.html;
    } else if (value === true || value === false) {
      dd.innerHTML = `<span class="${value ? 'yes' : 'no'}">${show(value, opts)}</span>`;
    } else {
      dd.textContent = show(value, opts);
    }
    container.append(dt, dd);
  });
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

/* ─────────────────────────── Health ─────────────────────────── */

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
      `Rule Engine ${c.rule_engine.version || c.rule_engine.status}`,
      `τFP=${health.thresholds.fingerprint} τMERT=${health.thresholds.embedding}`,
    ];
    dot.className = `dot ${health.status === 'ONLINE' ? 'dot-ok' : 'dot-warn'}`;
    text.textContent = health.status === 'ONLINE' ? 'Hệ thống sẵn sàng' : 'Hoạt động hạn chế';
    box.title = bits.join('\n');
  } catch (_) {
    dot.className = 'dot dot-bad';
    text.textContent = 'Không kết nối được máy chủ';
  }
}

/* ─────────────────────────── Màn 1: Upload ─────────────────────────── */

function pickFile(file) {
  state.file = file || null;
  $('#file-picked').hidden = !file;
  $('#analyze').disabled = !file;
  $('#upload-error').hidden = true;
  if (file) {
    $('#file-name').textContent = file.name;
    $('#file-size').textContent = humanSize(file.size);
  }
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
  if (!state.file) return;

  const body = new FormData();
  body.append('file', state.file);
  body.append('platform', $('#platform').value);
  body.append('commercial_use', $('#commercial_use').value);
  body.append('monetization', $('#monetization').value);

  $('#analyze').disabled = true;
  $('#upload-error').hidden = true;

  try {
    const response = await fetch(`${API}/analyze`, { method: 'POST', body });
    if (!response.ok) throw new Error(await apiError(response));

    const { job_id } = await response.json();
    state.jobId = job_id;
    $('#job-id').textContent = job_id;
    $('#processing-file').textContent = `· ${state.file.name}`;
    resetStages();
    showScreen('processing');
    pollJob();
  } catch (err) {
    const box = $('#upload-error');
    box.textContent = err.message;
    box.hidden = false;
  } finally {
    $('#analyze').disabled = !state.file;
  }
}

/* ─────────────────────────── Màn 2: Processing ─────────────────────────── */

function resetStages() {
  $$('#stage-list li').forEach((li) => {
    li.className = '';
    const ms = li.querySelector('.ms');
    if (ms) ms.remove();
  });
  $('#job-status').textContent = 'QUEUED';
}

/** Tô danh sách bước theo `stage` THẬT do backend báo. */
function paintStages(stage) {
  const current = STAGE_ORDER.indexOf(stage);
  $$('#stage-list li').forEach((li, i) => {
    if (current < 0) { li.className = ''; return; }
    li.className = i < current ? 'done' : (i === current ? 'running' : '');
  });
}

/** Kết thúc job: bước nào CHẠY THẬT thì đánh dấu xong kèm độ trễ đo được, bước
 *  nào bị bỏ qua thì nói rõ là bỏ qua — không tô "xong" cho việc chưa hề làm. */
function markAllStagesDone(result) {
  const identification = (result.evidence || {}).identification || {};
  const timings = identification.timings_ms || {};
  const stoppedAtStage1 = (result.match || {}).pipeline_stage === 'STAGE_1_CHROMAPRINT';
  const identified = Boolean((result.identity || {}).recording_id);

  $$('#stage-list li').forEach((li) => {
    const stage = li.dataset.stage;
    const key = STAGE_TIMING[stage];
    const value = key ? timings[key] : undefined;

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
    }

    li.className = skipped ? 'skipped' : 'done';

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
      showScreen('upload');
      const box = $('#upload-error');
      box.textContent = `[${job.error_code}] ${job.error_message}`;
      box.hidden = false;
      return;
    }
    state.timer = setTimeout(pollJob, POLL_MS);
  } catch (err) {
    stopPolling();
    showScreen('upload');
    const box = $('#upload-error');
    box.textContent = err.message;
    box.hidden = false;
  }
}

async function loadResult() {
  stopPolling();
  const response = await fetch(`${API}/results/${state.jobId}`);
  if (!response.ok) {
    const box = $('#upload-error');
    box.textContent = await apiError(response);
    box.hidden = false;
    showScreen('upload');
    return;
  }
  state.result = await response.json();
  markAllStagesDone(state.result);
  renderResult(state.result);
  renderEvidence(state.result);
  showScreen('result');
}

/* ─────────────────────────── Màn 3: Result ─────────────────────────── */

function renderResult(result) {
  const { identity = {}, match = {}, rights = {}, assessment = {} } = result;
  const risk = assessment.risk || 'UNKNOWN';

  const banner = $('#risk-banner');
  banner.className = `risk-banner risk-${risk}`;
  $('#risk-emoji').textContent = RISK_EMOJI[risk] || '⚪';
  $('#risk-level').textContent = `${risk} — ${RISK_TEXT[risk] || ''}`;
  $('#risk-condition').textContent = assessment.condition
    ? `Điều kiện: ${assessment.condition}` : '';

  dl($('#identity-list'), [
    ['Bài hát', identity.track],
    ['Nghệ sĩ', identity.artist],
    ['Loại khớp', null, {
      html: `<span class="pill">${show(match.type)}</span>`,
    }],
    ['Độ tin cậy nhận diện', match.confidence !== undefined && match.confidence !== null
      ? match.confidence.toFixed(4) : null],
    ['Tầng xử lý', match.pipeline_stage],
    ['recording_id', identity.recording_id, { mono: true }],
    ['composition_id', identity.composition_id, { mono: true }],
  ]);

  dl($('#rights-list'), [
    ['Giấy phép', rights.license],
    ['Trạng thái bản quyền', rights.copyright_status],
    ['Bắt buộc ghi nguồn', rights.attribution_required],
    ['Cho dùng thương mại', rights.commercial_use_allowed],
    ['Cho bật kiếm tiền', rights.monetization_allowed],
    ['Nguồn dữ liệu', rights.source],
    ['Xác minh lần cuối', rights.verified_at],
  ]);

  $('#recommendation').textContent = result.recommendation || '—';
  $('#decision-reason').textContent = assessment.reason
    ? `Lý do quyết định: ${assessment.reason}` : '';

  const confidences = [
    ['Độ tin cậy nhận diện', assessment.identity_confidence],
    ['Độ tin cậy dữ liệu quyền', assessment.rights_confidence],
    ['Độ tin cậy quyết định', assessment.decision_confidence],
  ];
  $('#confidences').innerHTML = confidences.map(([label, value]) => {
    const v = Number(value || 0);
    return `<div class="conf-row"><span>${label}</span>
      <span class="bar"><i style="width:${(v * 100).toFixed(0)}%"></i></span>
      <span class="mono">${v.toFixed(3)}</span></div>`;
  }).join('');

  $('#feedback-msg').hidden = true;
  $$('.feedback-buttons button').forEach((b) => b.setAttribute('aria-pressed', 'false'));
}

/* ─────────────────────────── Màn 4: Evidence ─────────────────────────── */

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
    ['Ngưỡng τFP', fingerprint.threshold],
    ['Kết luận tầng 1', fingerprint.match_type],
    ...(fpRan ? [] : [['Lý do tầng 1 không chạy',
      [fingerprint.reason_code, fingerprint.message].filter(Boolean).join(' — ') || null]]),
    ['Số fingerprint đã so', fingerprint.candidates_compared],
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
    ['Ngưỡng định danh tối thiểu', rules.min_identity_confidence],
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

  dl($('#ev-source'), [
    ['Nguồn metadata quyền', rightsRecord.source],
    ['Đường dẫn nguồn', null, {
      html: rightsRecord.source_url
        ? `<a href="${rightsRecord.source_url}" target="_blank" rel="noopener">${rightsRecord.source_url}</a>`
        : '—',
    }],
    ['Ngày xác minh', rightsRecord.verified_at],
    ['Phạm vi quyền', rightsRecord.rights_scope],
    ['Hiệu lực', rightsRecord.valid_from
      ? `${rightsRecord.valid_from} → ${show(rightsRecord.valid_until)}` : null],
    ['Lãnh thổ / nền tảng', rightsRecord.territory
      ? `${rightsRecord.territory} / ${show(rightsRecord.platform)}` : null],
    ['Tác phẩm (composition)', composition.title],
    ['Tác giả', composition.composer],
    // §2: PD của tác phẩm và PD của bản thu là hai trường ĐỘC LẬP
    ['PD của tác phẩm', composition.public_domain_status],
    ['PD của bản thu', rightsRecord.recording_public_domain],
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
        ['Dịch cao độ (OTI)', topCand.oti !== undefined ? `${topCand.oti} bán cung` : null],
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

/* ─────────────────────────── Phản hồi ─────────────────────────── */

async function sendFeedback(verdict, button) {
  const msg = $('#feedback-msg');
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
  }
  msg.hidden = false;
}

/* ─────────────────────────── Khởi động ─────────────────────────── */

function restart() {
  stopPolling();
  state.jobId = null;
  state.result = null;
  $('#file').value = '';
  pickFile(null);
  showScreen('upload');
}

document.addEventListener('DOMContentLoaded', () => {
  initUpload();
  checkHealth();

  $('#to-evidence').addEventListener('click', () => showScreen('evidence'));
  $('#back-to-result').addEventListener('click', () => showScreen('result'));
  $('#restart').addEventListener('click', restart);
  $('#restart2').addEventListener('click', restart);
  $('#cancel-poll').addEventListener('click', () => { stopPolling(); showScreen('upload'); });

  $$('.feedback-buttons button').forEach((button) =>
    button.addEventListener('click', () => sendFeedback(button.dataset.verdict, button)));

  $$('.steps li').forEach((li) => li.addEventListener('click', () => {
    const target = li.dataset.step;
    if (target === 'upload') return restart();
    if (!state.result) return;                       // chưa có kết quả thì không nhảy
    if (target === 'result' || target === 'evidence') showScreen(target);
  }));
});
