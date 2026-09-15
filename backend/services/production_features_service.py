"""
Đo ĐẶC TRƯNG SẢN XUẤT của một đoạn audio: nó nghe giống bản phát hành thương mại
được master chuyên nghiệp, hay giống bản thu nghiệp dư / demo?

Vì sao module này tồn tại tách khỏi bộ phân loại giấy phép. EXP-09 cho thấy giấy
phép KHÔNG học được từ âm thanh (bộ phân loại không vượt baseline khi gặp nghệ sĩ
mới). Nhưng có một thứ THẬT SỰ nằm trong tín hiệu và đo được: dấu vết của quy
trình hậu kỳ. Bản phát hành thương mại gần như luôn qua mastering — nén dải động
mạnh, đẩy loudness sát trần, giới hạn đỉnh — còn bản thu tại nhà thì không.

Điều PHẢI nói rõ và không được quên khi đọc kết quả:

  * Đây KHÔNG PHẢI chỉ báo bản quyền. Rất nhiều bản nhạc Creative Commons cũng
    được master chuyên nghiệp, và rất nhiều bản thu thương mại cố tình giữ dải
    động rộng (nhạc cổ điển, jazz thu mộc).
  * Nó chỉ trả lời một câu hẹp: "bản thu này có dấu vết mastering thương mại
    không". Suy từ đó ra tình trạng pháp lý là đúng loại suy diễn mà §2 cấm.
  * Vì vậy điểm số ở đây chỉ đi vào phần Evidence, KHÔNG được phép đổi mức rủi ro.

Các chỉ số đều là đại lượng vật lý chuẩn trong xử lý âm thanh, tính thẳng từ tín
hiệu chứ không qua model nào — nên chúng không có "độ chính xác" để nghi ngờ, chỉ
có cách diễn giải cần thận trọng.
"""
import numpy as np

# Ngưỡng diễn giải, lấy theo thông lệ mastering chứ không phải hiệu chỉnh trên
# dữ liệu của dự án. Ghi rõ nguồn gốc để không ai tưởng đây là số đã tối ưu:
#   * RMS -14 dBFS xấp xỉ mức loudness mà các nền tảng phát nhạc chuẩn hoá về.
#   * Crest factor dưới 10 dB là dấu hiệu nén dải động mạnh, đặc trưng của
#     "loudness war" trong nhạc phát hành thương mại.
LOUD_RMS_DBFS = -14.0
COMPRESSED_CREST_DB = 10.0
CLIPPING_RATIO_HIGH = 0.001      # 0,1% mẫu chạm trần là đã đáng chú ý


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-10)))


def extract_production_features(audio, sample_rate: int) -> dict:
    """
    Trả các chỉ số sản xuất đo được từ tín hiệu mono.

    Trả None nếu tín hiệu quá ngắn — thà không có số còn hơn có một con số tính
    trên vài mẫu rồi bị đọc như thật.
    """
    if audio is None:
        return None
    audio = np.asarray(audio, dtype="float32").reshape(-1)
    if len(audio) < sample_rate:          # dưới 1 giây
        return None

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak_db, rms_db = _db(peak), _db(rms)

    # Crest factor = đỉnh / RMS. Càng nhỏ nghĩa là dải động càng bị nén.
    crest_db = peak_db - rms_db

    # Tỉ lệ mẫu sát trần: dấu vết của limiter đẩy kịch hoặc clipping thật
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))

    # Dải động ngắn hạn: độ lệch chuẩn của RMS theo từng khung 50 ms. Bản đã nén
    # mạnh thì độ to gần như không đổi theo thời gian.
    frame = max(int(0.05 * sample_rate), 1)
    usable = len(audio) - (len(audio) % frame)
    frames = audio[:usable].reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-10))
    loud_frames = frame_db[frame_db > (np.max(frame_db) - 40.0)]  # bỏ khoảng lặng
    dynamic_range_db = float(np.percentile(loud_frames, 95) -
                             np.percentile(loud_frames, 10)) if len(loud_frames) else 0.0

    return {
        "peak_dbfs": round(peak_db, 2),
        "rms_dbfs": round(rms_db, 2),
        "crest_factor_db": round(crest_db, 2),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "clipping_ratio": round(clipping_ratio, 6),
        "duration_s": round(len(audio) / sample_rate, 2),
    }


def production_polish(features: dict) -> dict:
    """
    Gộp các chỉ số thành một điểm 0..1 kèm DIỄN GIẢI BẰNG LỜI của từng thành phần.

    Trả kèm `contributions` chứ không chỉ một con số: §2 cấm dùng một điểm duy
    nhất làm kết luận, và ở đây người đọc phải thấy được điểm số đến từ đâu để
    tự phản bác được.
    """
    if not features:
        return None

    contributions = []

    # 1. Loudness: càng gần/vượt mức chuẩn hoá của nền tảng càng giống bản master
    loudness = float(np.clip((features["rms_dbfs"] - (LOUD_RMS_DBFS - 12.0)) / 12.0,
                             0.0, 1.0))
    contributions.append({
        "yeu_to": "loudness",
        "gia_tri": features["rms_dbfs"],
        "diem": round(loudness, 3),
        "giai_thich": (f"RMS {features['rms_dbfs']} dBFS so với mức tham chiếu "
                       f"{LOUD_RMS_DBFS} dBFS của các nền tảng phát nhạc"),
    })

    # 2. Nén dải động: crest factor thấp = nén mạnh = dấu vết mastering
    compression = float(np.clip((20.0 - features["crest_factor_db"]) / 12.0, 0.0, 1.0))
    contributions.append({
        "yeu_to": "nen_dai_dong",
        "gia_tri": features["crest_factor_db"],
        "diem": round(compression, 3),
        "giai_thich": (f"Crest factor {features['crest_factor_db']} dB; dưới "
                       f"{COMPRESSED_CREST_DB} dB là dấu hiệu nén mạnh"),
    })

    # 3. Dải động ngắn hạn hẹp cũng chỉ về hậu kỳ nặng tay
    consistency = float(np.clip((18.0 - features["dynamic_range_db"]) / 14.0, 0.0, 1.0))
    contributions.append({
        "yeu_to": "on_dinh_do_to",
        "gia_tri": features["dynamic_range_db"],
        "diem": round(consistency, 3),
        "giai_thich": (f"Dải động ngắn hạn {features['dynamic_range_db']} dB — "
                       f"càng hẹp thì độ to càng đều theo thời gian"),
    })

    score = float(np.mean([loudness, compression, consistency]))
    if score >= 0.66:
        label = "co_dau_vet_mastering_thuong_mai"
    elif score >= 0.4:
        label = "khong_ket_luan_duoc"
    else:
        label = "giong_ban_thu_moc_hoac_nghiep_du"

    return {
        "polish_score": round(score, 3),
        "nhan": label,
        "contributions": contributions,
        "clipping_dang_chu_y": features["clipping_ratio"] >= CLIPPING_RATIO_HIGH,
        # Hai trường dưới đây đi kèm điểm số ở MỌI nơi nó xuất hiện.
        "khong_phai_chi_bao_ban_quyen": True,
        "canh_bao": (
            "Điểm này đo dấu vết MASTERING, không đo tình trạng bản quyền. Nhạc "
            "Creative Commons cũng có thể được master chuyên nghiệp, và nhiều bản "
            "thu thương mại (cổ điển, jazz mộc) cố ý giữ dải động rộng. Không "
            "được dùng để suy ra quyền sử dụng."
        ),
    }


def analyze(audio, sample_rate: int) -> dict:
    """Tiện ích: đo đặc trưng rồi gộp điểm trong một lần gọi."""
    features = extract_production_features(audio, sample_rate)
    if not features:
        return None
    return {"features": features, "assessment": production_polish(features)}
