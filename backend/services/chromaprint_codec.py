"""
Bộ giải nén fingerprint Chromaprint viết thuần Python.

Vì sao cần: `acoustid.compare_fingerprints` phải giải nén fingerprint base64
đang lưu trong DB, việc đó đòi thư viện native `libchromaprint`. Trên Windows
x64 không có gói conda-forge/pip nào cung cấp thư viện này (chỉ có binary
`fpcalc.exe`). Module này gỡ bỏ hoàn toàn phụ thuộc native đó.

Thuật toán bám sát `fingerprint_decompressor.cpp` của Chromaprint:
  - 4 byte header: [0] = algorithm, [1..3] = số phần tử (big-endian 24 bit)
  - luồng "normal bits": mỗi phần tử 3 bit, giá trị 0 = kết thúc một số nguyên
  - căn biên byte, rồi tới luồng "exception bits": 5 bit cho mỗi giá trị = 7
  - mỗi số nguyên được dựng từ vị trí các bit bật, sau đó XOR với số liền trước

Tính đúng đắn được kiểm chứng bằng `tests/test_chromaprint_codec.py`: giải nén
chuỗi base64 của `fpcalc` phải ra đúng dãy số nguyên của `fpcalc -raw`.
"""
import base64

NORMAL_BITS = 3
EXCEPTION_BITS = 5
MAX_NORMAL_VALUE = (1 << NORMAL_BITS) - 1  # 7


class InvalidFingerprintError(ValueError):
    """Chuỗi fingerprint không đúng định dạng Chromaprint."""


class _BitReader:
    """Đọc chuỗi bit theo thứ tự LSB-first, giống BitStringReader của C++."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0
        self._buffer = 0
        self._buffer_size = 0

    def read(self, bits: int) -> int:
        if self._buffer_size < bits:
            if self._pos < len(self._data):
                self._buffer |= self._data[self._pos] << self._buffer_size
                self._pos += 1
                self._buffer_size += 8
            else:
                raise InvalidFingerprintError("Hết dữ liệu khi đang đọc fingerprint")
        result = self._buffer & ((1 << bits) - 1)
        self._buffer >>= bits
        self._buffer_size -= bits
        return result

    def reset(self) -> None:
        """Bỏ phần bit lẻ còn lại, căn về biên byte."""
        self._buffer = 0
        self._buffer_size = 0


def decode_base64(fp) -> bytes:
    """Giải mã base64 URL-safe không padding của Chromaprint."""
    if isinstance(fp, bytes):
        text = fp.decode("ascii")
    else:
        text = str(fp)
    text = text.strip()
    padding = "=" * (-len(text) % 4)
    try:
        return base64.urlsafe_b64decode(text + padding)
    except Exception as e:
        raise InvalidFingerprintError(f"Không giải mã được base64: {e}") from e


def decode_fingerprint(fp) -> tuple[list[int], int]:
    """
    Giải nén fingerprint -> (danh sách số nguyên 32-bit, mã thuật toán).

    Chữ ký trả về giống `chromaprint.decode_fingerprint` để có thể thay thế
    trực tiếp thư viện native.
    """
    data = decode_base64(fp)
    if len(data) < 4:
        raise InvalidFingerprintError("Fingerprint ngắn hơn 4 byte header")

    algorithm = data[0]
    num_values = (data[1] << 16) | (data[2] << 8) | data[3]
    if num_values == 0:
        return [], algorithm

    reader = _BitReader(data[4:])

    # 1. Luồng normal bits: đọc tới khi đủ num_values số kết thúc bằng bit 0
    bits, found = [], 0
    while found < num_values:
        bit = reader.read(NORMAL_BITS)
        if bit == 0:
            found += 1
        bits.append(bit)

    # 2. Luồng exception bits (căn biên byte trước khi đọc)
    reader.reset()
    for i, bit in enumerate(bits):
        if bit == MAX_NORMAL_VALUE:
            bits[i] = bit + reader.read(EXCEPTION_BITS)

    # 3. Dựng lại số nguyên: vị trí bit bật là tổng tích luỹ, rồi XOR delta
    result = [0] * num_values
    i, last_bit, value = 0, 0, 0
    for bit in bits:
        if bit == 0:
            result[i] = value ^ result[i - 1] if i > 0 else value
            value, last_bit = 0, 0
            i += 1
            continue
        bit += last_bit
        last_bit = bit
        value |= 1 << (bit - 1)

    return result, algorithm


# fpcalc mặc định chỉ đọc 120 giây đầu, và sinh khoảng 7–8 hash cho mỗi giây audio.
FPCALC_DEFAULT_LENGTH_S = 120.0
PLAUSIBLE_HASHES_PER_SECOND = (4.0, 12.0)
# Bộ lọc của Chromaprint "ăn" mất vài chục khung ở hai đầu, nên đoạn ngắn cho tỉ lệ
# hash/giây thấp hơn hẳn — nới biên một khoảng cố định để không loại nhầm chúng.
HASH_COUNT_SLACK = 20


def is_plausible_fingerprint(fp, duration) -> bool:
    """
    Chuỗi này có thể là đầu ra thật của fpcalc cho một đoạn dài `duration` giây không.

    Kiểm tra chính nội dung chứ không tin nhãn `algorithm`: một chuỗi không sinh
    từ audio thật hoặc không giải nén được, hoặc cho số hash quá ít so với độ dài.
    """
    try:
        values, _ = decode_fingerprint(fp)
        seconds = min(float(duration), FPCALC_DEFAULT_LENGTH_S)
    except (InvalidFingerprintError, TypeError, ValueError):
        return False
    if seconds <= 0:
        return False
    low, high = PLAUSIBLE_HASHES_PER_SECOND
    return low * seconds - HASH_COUNT_SLACK <= len(values) <= high * seconds + HASH_COUNT_SLACK
