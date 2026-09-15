# Claude Code Automated Hooks (`.claude/hooks`)

Thư mục này chứa các hook tự động hóa quy trình làm việc cho dự án **Hệ thống AI Phân loại Nhạc Bản quyền**.

---

## 1. Cơ chế Hoạt động (How It Works)

Hệ thống hook được cấu hình tại [.claude/settings.json](file:///d:/PycharmProjects/AMR_advanced/.claude/settings.json) dưới sự kiện **`PostToolUse`**:

```json
"hooks": {
  "PostToolUse": [
    {
      "matcher": "Edit|Write",
      "hooks": [
        {
          "type": "command",
          "command": "python .claude/hooks/format_and_lint.py"
        }
      ]
    }
  ]
}
```

Mỗi khi Claude Code thực hiện một thao tác ghi tệp (`Write`) hoặc chỉnh sửa tệp (`Edit`):
1. **Bắt sự kiện tự động**: Claude Code truyền thông tin file vừa sửa (dưới dạng JSON payload qua `stdin`) vào script [format_and_lint.py](file:///d:/PycharmProjects/AMR_advanced/.claude/hooks/format_and_lint.py).
2. **Kiểm tra Cú pháp (Syntax Validation)**:
   - Sử dụng module `ast` của Python để phân tích cú pháp mã nguồn.
   - Nếu phát hiện `SyntaxError`, hook trả về **`exit code 2`** kèm thông báo lỗi chi tiết qua `stderr`.
   - **Tự động sửa lỗi (Self-Healing)**: Khi nhận exit code 2, Claude Code sẽ tự động bắt lấy thông báo lỗi từ hook và viết lại đoạn code bị lỗi ngay lập tức mà người dùng không cần nhắc.
3. **Tự động Chuẩn hóa & Format (Auto-Format & Auto-Fix)**:
   - Tự động gọi `ruff check --fix` để sửa các lỗi import thừa, sắp xếp import, chuẩn hóa biến.
   - Tự động gọi `ruff format` để format mã nguồn đúng chuẩn PEP 8.
4. **Kiểm tra Cú pháp Cấu hình (JSON / YAML)**:
   - Tự động kiểm tra file `.json` qua `json.load`.
   - Tự động kiểm tra file `.yaml` / `.yml` qua `yaml.safe_load`.
5. **Cảnh báo Vi phạm Nguyên tắc Dự án (Guardrails)**:
   - Phát hiện các anti-pattern như: sử dụng `CLAP` làm copyright classifier, cấu trúc `except: pass` nuốt lỗi ngầm.

---

## 2. Kiểm thử Thủ công Hook Script

Bạn có thể tự chạy hook trên bất kỳ file nào bằng lệnh:

```powershell
python .claude/hooks/format_and_lint.py đường_dẫn_tới_file.py
```

- Nếu file hợp lệ: Mã thoát `0`.
- Nếu file có lỗi cú pháp nghiêm trọng: Mã thoát `2`.

---

## 3. Tích hợp với Git Hooks (Tùy chọn)

Nếu bạn muốn chạy hook này trước mỗi lần `git commit`, bạn có thể tạo file `.git/hooks/pre-commit`:

```bash
#!/bin/sh
# Chạy hook kiểm tra trước khi commit
git diff --cached --name-only | while read file; do
    python .claude/hooks/format_and_lint.py "$file"
    if [ $? -eq 2 ]; then
        echo "Commit bị từ chối do lỗi cú pháp trong $file"
        exit 1
    fi
done
```
