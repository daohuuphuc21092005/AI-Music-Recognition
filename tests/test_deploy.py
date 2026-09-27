"""Kiểm thử các script triển khai lên Hugging Face Spaces (không cần mạng, không cần CSDL)."""
from pathlib import Path

import pytest

from backend import config
from deploy import build_space_bundle, fetch_runtime_data, setup_neon_db


def test_goi_space_chi_chua_phan_can_chay(tmp_path):
    files = build_space_bundle.build(tmp_path / "bundle")

    # Hugging Face tìm Dockerfile và thẻ Space ở gốc
    assert "Dockerfile" in files and "README.md" in files
    readme = (tmp_path / "bundle" / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n") and "sdk: docker" in readme and "app_port: 7860" in readme
    assert "backend/main.py" in files and "frontend/index.html" in files
    assert "configs/rules_v1.yaml" in files and "deploy/fetch_runtime_data.py" in files

    # Không mang theo dữ liệu, test, thí nghiệm, bí mật hay file biên dịch
    for prefix in ("data/", "tests/", "experiments/", ".claude/", "scripts/"):
        assert not any(f.startswith(prefix) for f in files), prefix
    assert not any(Path(f).name.startswith(".env") for f in files)
    assert not any("__pycache__" in f for f in files)
    assert b"\r\n" not in (tmp_path / "bundle" / "deploy/space/start.sh").read_bytes()


def test_goi_space_khong_xoa_nham_thu_muc_repo():
    for bad in (build_space_bundle.ROOT, build_space_bundle.ROOT / "backend",
                build_space_bundle.ROOT / "deploy", build_space_bundle.ROOT.parent):
        with pytest.raises(ValueError):
            build_space_bundle.build(bad)


def test_tai_chi_muc_thieu_file_tuy_chon_van_dat(tmp_path, monkeypatch):
    sources = {}
    for name in ("mert_faiss.index", "faiss_id_map.json"):
        path = tmp_path / "hub" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        sources[name] = path
    targets = {name: tmp_path / "data" / name for name in sources}
    monkeypatch.setattr(fetch_runtime_data, "RUNTIME_FILES", (
        ("mert_faiss.index", str(targets["mert_faiss.index"]), True),
        ("faiss_id_map.json", str(targets["faiss_id_map.json"]), True),
        ("cover_descriptors.npy", str(tmp_path / "data" / "cover.npy"), False),
    ))

    def download(repo_id, filename, token):
        if filename not in sources:
            raise FileNotFoundError(filename)
        return str(sources[filename])

    assert fetch_runtime_data.fetch("u/data", None, download=download) == []
    assert targets["mert_faiss.index"].read_bytes() == b"mert_faiss.index"

    sources.pop("faiss_id_map.json")
    targets["faiss_id_map.json"].unlink()
    assert fetch_runtime_data.fetch("u/data", None, download=download) == ["faiss_id_map.json"]


def test_file_chi_muc_tai_ve_dung_duong_dan_server_doc():
    targets = {name: target for name, target, _ in fetch_runtime_data.RUNTIME_FILES}
    assert targets["mert_faiss.index"] == config.FAISS_INDEX_PATH
    assert targets["faiss_id_map.json"] == config.FAISS_ID_MAP_PATH


def test_url_role_ung_dung_giu_may_chu_va_tham_so():
    owner = "postgresql://neondb_owner:bi-mat@ep-x-pooler.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    url = setup_neon_db.app_role_url(owner, "p@ss/w:rd")
    assert url.startswith("postgresql://music_ai_app:p%40ss%2Fw%3Ard@ep-x-pooler.aws.neon.tech/neondb?")
    assert "sslmode=require" in url and "channel_binding=require" in url
    assert "bi-mat" not in url and "neondb_owner" not in url


def test_role_sql_dung_ten_csdl_va_dat_lai_mat_khau():
    sql = setup_neon_db.role_sql("neondb", "o'k")
    assert 'GRANT CONNECT ON DATABASE "neondb"' in sql
    assert "<MAT_KHAU_APP_USER>" not in sql
    assert "PASSWORD 'o''k'" in sql
    assert sql.rstrip().endswith("ALTER ROLE music_ai_app WITH LOGIN PASSWORD 'o''k';")
    assert "REVOKE CREATE ON SCHEMA public FROM music_ai_app" in sql


def test_khong_chep_de_len_chinh_csdl_nguon():
    assert setup_neon_db.same_database("postgresql://a:x@127.0.0.1:5433/music_rights_ai",
                                       "postgresql://b:y@127.0.0.1:5433/music_rights_ai?sslmode=disable")
    assert not setup_neon_db.same_database("postgresql://a@127.0.0.1:5433/music_rights_ai",
                                           "postgresql://a@ep-x.neon.tech/neondb")


def test_ghi_file_env_thay_dong_cu(tmp_path):
    env = tmp_path / ".env.neon"
    env.write_text("NEON_OWNER_URL=postgresql://x\nSPACE_DATABASE_URL=cu\n", encoding="utf-8")
    setup_neon_db.write_env_value(env, "SPACE_DATABASE_URL", "moi")
    assert setup_neon_db.read_env_file(env) == {"NEON_OWNER_URL": "postgresql://x",
                                                "SPACE_DATABASE_URL": "moi"}
