"""
Điểm truy cập DUY NHẤT tới PostgreSQL.

Trước đây chuỗi kết nối bị hardcode lặp lại ở 8 file khác nhau, khiến
docker-compose truyền biến DATABASE_URL nhưng code không hề đọc.
Mọi module giờ import engine/SessionLocal/get_db từ đây.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency: mở session cho mỗi request và luôn đóng lại."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_connection() -> tuple[bool, str]:
    """Kiểm tra kết nối DB, trả (ok, thông điệp ngắn). Không raise."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "CONNECTED"
    except Exception as e:
        return False, f"UNAVAILABLE: {type(e).__name__}"
