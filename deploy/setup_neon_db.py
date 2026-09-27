"""
Chuẩn bị CSDL Neon cho bản chạy trên web: tạo bảng, chép dữ liệu tham chiếu từ CSDL
trên máy này, tạo role ứng dụng quyền tối thiểu.

    # 1. Dán chuỗi kết nối Neon (role chủ, lấy ở nút "Connect" của Neon) vào file
    #    .env.neon ở gốc repo — file này đã bị .gitignore chặn (mẫu .env.*):
    #        NEON_OWNER_URL=postgresql://neondb_owner:...@ep-....neon.tech/neondb?sslmode=require
    # 2. Chạy:
    python deploy/setup_neon_db.py

Việc script làm, theo thứ tự:
  1. `init_db.py --schema-only` trên Neon bằng role chủ (tạo bảng, không nạp CSV).
  2. Chép compositions, recordings, rights, fingerprints từ CSDL nguồn (mặc định là
     DATABASE_URL trong .env) sang Neon bằng COPY. KHÔNG chép embeddings (586 MB, server
     không đọc bảng này khi chạy, chỉ mục FAISS đã chứa đủ) và KHÔNG chép job, phản hồi,
     sổ bài chưa nhận diện: bản web bắt đầu với lịch sử trống.
  3. Chạy scripts/sql/create_app_role.sql trên Neon với một mật khẩu ngẫu nhiên, rồi ghi
     SPACE_DATABASE_URL (role music_ai_app) vào .env.neon. Chuỗi này dán vào secret
     SPACE_DATABASE_URL của repo GitHub. Mật khẩu không bao giờ được in ra màn hình.

Bảng đích đã có dữ liệu thì script dừng, trừ khi thêm --replace (xoá dữ liệu cũ TRÊN
NEON rồi chép lại). CSDL nguồn chỉ bị ĐỌC.
"""
import argparse
import os
import re
import secrets
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
ENV_NEON = ROOT / ".env.neon"
ROLE_SQL = ROOT / "scripts" / "sql" / "create_app_role.sql"
APP_ROLE = "music_ai_app"
# Thứ tự theo khoá ngoại: bảng được tham chiếu nạp trước
TABLES = ("compositions", "recordings", "rights", "fingerprints")


def read_env_file(path: Path) -> dict:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def write_env_value(path: Path, key: str, value: str) -> None:
    """Đặt KEY=value trong file .env (thay dòng cũ nếu có), giữ nguyên các dòng khác."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [line for line in lines if not line.strip().startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def with_query(url: str, **params) -> str:
    """Thêm tham số vào query của URL nếu URL chưa có tham số đó."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    for key, value in params.items():
        query.setdefault(key, value)
    return urlunsplit(parts._replace(query=urlencode(query)))


def same_database(url_a: str, url_b: str) -> bool:
    """Cùng máy chủ, cổng và tên CSDL (bỏ qua người dùng, mật khẩu, tham số)."""
    def key(url):
        parts = urlsplit(url)
        return (parts.hostname or "").lower(), parts.port or 5432, parts.path.lstrip("/")
    return key(url_a) == key(url_b)


def app_role_url(owner_url: str, password: str) -> str:
    """Cùng máy chủ, CSDL và tham số với URL của role chủ, nhưng đăng nhập bằng music_ai_app."""
    parts = urlsplit(owner_url)
    host = parts.hostname + (f":{parts.port}" if parts.port else "")
    netloc = f"{APP_ROLE}:{quote(password, safe='')}@{host}"
    return urlunsplit(parts._replace(netloc=netloc))


def role_sql(database: str, password: str) -> str:
    """create_app_role.sql với tên CSDL và mật khẩu thật (Neon mặc định là `neondb`)."""
    sql = ROLE_SQL.read_text(encoding="utf-8")
    sql = sql.replace("<MAT_KHAU_APP_USER>", password.replace("'", "''"))
    sql = re.sub(r"(GRANT CONNECT ON DATABASE )music_rights_ai", rf'\1"{database}"', sql)
    # Role có sẵn từ lần chạy trước thì khối CREATE ROLE bỏ qua: đặt lại mật khẩu cho khớp
    return sql + f"\nALTER ROLE {APP_ROLE} WITH LOGIN PASSWORD '{password.replace(chr(39), chr(39) * 2)}';\n"


def common_columns(src_cur, dst_cur, table: str) -> list:
    query = ("SELECT column_name FROM information_schema.columns "
             "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position")
    src_cur.execute(query, (table,))
    source = {row[0] for row in src_cur.fetchall()}
    dst_cur.execute(query, (table,))
    return [row[0] for row in dst_cur.fetchall() if row[0] in source]


def copy_table(src, dst, table: str) -> int:
    with src.cursor() as src_cur, dst.cursor() as dst_cur:
        columns = common_columns(src_cur, dst_cur, table)
        if not columns:
            raise RuntimeError(f"Bảng {table} không có cột chung giữa nguồn và Neon")
        column_list = ", ".join(f'"{c}"' for c in columns)
        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode="w+b") as buffer:
            src_cur.copy_expert(f"COPY (SELECT {column_list} FROM {table}) TO STDOUT", buffer)
            buffer.seek(0)
            dst_cur.copy_expert(f"COPY {table} ({column_list}) FROM STDIN", buffer)
        dst_cur.execute(f"SELECT COUNT(*) FROM {table}")
        return dst_cur.fetchone()[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--replace", action="store_true",
                        help="bảng trên Neon đã có dữ liệu thì xoá đi rồi chép lại")
    parser.add_argument("--source-url", default=None,
                        help="CSDL nguồn (mặc định: DATABASE_URL trong .env)")
    parser.add_argument("--env-file", default=str(ENV_NEON),
                        help="file chứa NEON_OWNER_URL, nơi ghi SPACE_DATABASE_URL (mặc định .env.neon)")
    args = parser.parse_args()
    env_file = Path(args.env_file)

    owner_url = os.environ.get("NEON_OWNER_URL") or read_env_file(env_file).get("NEON_OWNER_URL")
    if not owner_url:
        print(f"Chưa có NEON_OWNER_URL. Dán chuỗi kết nối Neon vào {env_file.name}:\n"
              "  NEON_OWNER_URL=postgresql://...neon.tech/neondb?sslmode=require")
        return 2
    source_url = (args.source_url or os.environ.get("SOURCE_DATABASE_URL")
                  or read_env_file(ROOT / ".env").get("DATABASE_URL"))
    if not source_url:
        print("Không tìm thấy CSDL nguồn: đặt DATABASE_URL trong .env hoặc dùng --source-url")
        return 2
    if same_database(owner_url, source_url):
        print("NEON_OWNER_URL và CSDL nguồn là cùng một CSDL: dừng để tránh chép đè lên chính nó")
        return 2

    # URL cho role ứng dụng giữ đúng tham số người dùng dán vào (sslmode, channel_binding…)
    given_owner_url = owner_url
    # libpq trên Windows thử GSSAPI trước và có thể hỏng; Neon dùng SSL
    owner_url = with_query(owner_url, gssencmode="disable", sslmode="require")
    source_url = with_query(source_url, gssencmode="disable")

    import psycopg2

    # 1. Tạo bảng trên Neon: init_db đọc DATABASE_URL lúc import, biến môi trường thắng .env
    print("1/3 Tạo bảng trên Neon (init_db.py --schema-only)")
    os.environ["DATABASE_URL"] = owner_url
    sys.path.insert(0, str(ROOT))
    import init_db
    init_db.init_database(schema_only=True)

    src = psycopg2.connect(source_url)
    dst = psycopg2.connect(owner_url)
    try:
        src.set_session(readonly=True)
        # 2. Chép dữ liệu tham chiếu
        print("2/3 Chép dữ liệu tham chiếu")
        with dst.cursor() as cur:
            nonempty = []
            for table in TABLES:
                cur.execute(f"SELECT EXISTS (SELECT 1 FROM {table})")
                if cur.fetchone()[0]:
                    nonempty.append(table)
        if nonempty and not args.replace:
            print(f"Neon đã có dữ liệu ở: {', '.join(nonempty)}. Chạy lại với --replace để chép đè.")
            return 1
        if nonempty:
            with dst.cursor() as cur:
                cur.execute(f"TRUNCATE {', '.join(TABLES)} CASCADE")
        for table in TABLES:
            count = copy_table(src, dst, table)
            print(f"   {table}: {count:,} dòng")
        dst.commit()

        # 3. Role ứng dụng quyền tối thiểu
        print(f"3/3 Tạo role {APP_ROLE}")
        with dst.cursor() as cur:
            cur.execute("SELECT current_database()")
            database = cur.fetchone()[0]
        password = secrets.token_urlsafe(24)
        with dst.cursor() as cur:
            cur.execute(role_sql(database, password))
        dst.commit()
    finally:
        src.close()
        dst.close()

    write_env_value(env_file, "SPACE_DATABASE_URL", app_role_url(given_owner_url, password))
    print(f"Xong. SPACE_DATABASE_URL (role {APP_ROLE}) đã ghi vào {env_file.name}; "
          "dán giá trị đó vào secret SPACE_DATABASE_URL của repo GitHub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
