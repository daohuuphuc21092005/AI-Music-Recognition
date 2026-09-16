"""
Quy tắc gắn nhãn trung thực cho dữ liệu nguồn (scripts/process_all_datasets.py).

Các hàm này quyết định một bản ghi được trình bày là "đã xác minh" hay "mô phỏng" —
tức quyết định Rule Engine có dám đưa ra kết luận về quyền hay phải trả UNKNOWN.
Trước khi có chúng, 100.000 bản ghi tổng hợp mang nguồn "Spotify API / Content ID"
và metadata_verified=True, nên không cơ chế phạt nào nhận ra đó là dữ liệu bịa.
"""
from scripts.process_all_datasets import (
    RECORDING_PD_MIN_AGE_YEARS,
    creator_music_flags,
    metadata_is_verified,
    recording_pd_plausible,
    spotify_rights_source,
)


def test_chi_fma_la_metadata_da_xac_minh():
    assert metadata_is_verified("FMA")
    for source in ("dataset_G_vietnam_100k_api", "MTG-Jamendo", "Spotify Web API",
                   "YouTube Audio Library", "Creator Music", ""):
        assert not metadata_is_verified(source), source


def test_ban_thu_moi_phat_hanh_khong_the_la_pd():
    cutoff = 2026 - RECORDING_PD_MIN_AGE_YEARS
    assert not recording_pd_plausible(2025, now_year=2026)
    assert not recording_pd_plausible(cutoff + 1, now_year=2026)
    assert recording_pd_plausible(cutoff, now_year=2026)
    # Giá trị đọc từ CSV có thể là "1950.0" hoặc rỗng
    assert recording_pd_plausible("1950.0", now_year=2026)
    assert not recording_pd_plausible("", now_year=2026)
    assert not recording_pd_plausible(None, now_year=2026)


def test_co_creator_music_loai_tru_nhau():
    for share in (True, False, "True", "False", "nan", "", None, float("nan")):
        purchased, agreed = creator_music_flags("CREATOR_MUSIC", share)
        assert purchased != agreed, share


def test_nan_khong_bi_doc_thanh_co_chia_doanh_thu():
    """bool(NaN) là True — chính lỗi đó từng bật cờ chia doanh thu cho cột trống."""
    assert creator_music_flags("CREATOR_MUSIC", float("nan")) == (True, False)
    assert creator_music_flags("CC_BY", float("nan")) == (False, False)
    assert creator_music_flags("CC_BY", "True") == (False, True)


def test_nhan_spotify_khong_con_nan():
    assert spotify_rights_source("nan") == "Spotify API / không rõ hãng phát hành"
    assert spotify_rights_source(None) == "Spotify API / không rõ hãng phát hành"
    assert spotify_rights_source("  ") == "Spotify API / không rõ hãng phát hành"
    assert spotify_rights_source("Sony Music") == "Spotify API / Sony Music"
