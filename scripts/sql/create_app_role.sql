-- =============================================================================
-- Script tạo role ứng dụng tối thiểu (Principle of Least Privilege)
--
-- Lý do: Mặc định backend kết nối bằng superuser `postgres`. Khi ứng dụng bị tấn
-- công SQL Injection hoặc lộ thông tin kết nối, kẻ tấn công có quyền huỷ diệt toàn
-- bộ CSDL hoặc gọi lệnh hệ thống.
--
-- Role `music_ai_app` chỉ có quyền tối thiểu cần thiết để hệ thống hoạt động:
--   - Đọc/Ghi/Cập nhật: jobs, analysis_results, feedback
--   - Chỉ đọc (SELECT): recordings, compositions, rights, fingerprints, embeddings
--   - Tuyệt đối KHÔNG có quyền DROP, TRUNCATE, ALTER TABLE.
--
-- CÁCH DÙNG:
-- 1. Chạy script này bằng user postgres:
--    psql -U postgres -d music_rights_ai -f scripts/sql/create_app_role.sql
-- 2. Đổi DATABASE_URL trong file .env:
--    DATABASE_URL=postgresql://music_ai_app:<MAT_KHAU_APP_USER>@localhost:5432/music_rights_ai
-- =============================================================================

-- 1. Tạo role nếu chưa tồn tại (thay đổi mật khẩu phù hợp)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'music_ai_app') THEN
        CREATE ROLE music_ai_app WITH LOGIN PASSWORD '<MAT_KHAU_APP_USER>';
    END IF;
END
$$;

-- 2. Cấp quyền kết nối CSDL và sử dụng schema
GRANT CONNECT ON DATABASE music_rights_ai TO music_ai_app;
GRANT USAGE ON SCHEMA public TO music_ai_app;

-- 3. Cấp quyền SELECT trên các bảng danh mục & tham chiếu (Read-Only)
GRANT SELECT ON TABLE recordings TO music_ai_app;
GRANT SELECT ON TABLE compositions TO music_ai_app;
GRANT SELECT ON TABLE rights TO music_ai_app;
GRANT SELECT ON TABLE fingerprints TO music_ai_app;
GRANT SELECT ON TABLE embeddings TO music_ai_app;
GRANT SELECT ON TABLE test_queries TO music_ai_app;

-- 4. Cấp quyền SELECT, INSERT, UPDATE trên các bảng nghiệp vụ runtime
GRANT SELECT, INSERT, UPDATE ON TABLE jobs TO music_ai_app;
GRANT SELECT, INSERT, UPDATE ON TABLE analysis_results TO music_ai_app;
GRANT SELECT, INSERT, UPDATE ON TABLE feedback TO music_ai_app;

-- 5. Cấp quyền trên các Sequences (cho auto-increment ID nếu có)
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO music_ai_app;

-- 6. Đảm bảo các bảng tạo mới trong tương lai tuân thủ quyền mặc định
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO music_ai_app;

-- 7. Thu hồi các quyền nguy hiểm
REVOKE CREATE ON SCHEMA public FROM music_ai_app;
