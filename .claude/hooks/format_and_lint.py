#!/usr/bin/env python3
"""
Hook Script: format_and_lint.py
Hook tự động hóa quy trình kiểm tra, sửa lỗi và format code cho Claude Code.

Hoạt động:
1. Nhận dữ liệu sự kiện từ stdin (Claude Code PostToolUse JSON payload).
2. Kiểm tra cú pháp (Syntax check) bằng AST: nếu lỗi cú pháp -> exit 2 (Claude Code sẽ nhận diện và tự sửa lỗi).
3. Tự động sửa lỗi linting & format mã nguồn với ruff (nếu có trong môi trường).
4. Kiểm tra cú pháp file .json và .yaml/.yml.
5. Kiểm tra các nguyên tắc bất biến (Anti-patterns của dự án).
"""

import sys
import json
import os
import ast
import subprocess

def log_err(msg: str):
    sys.stderr.write(f"[HOOK ERROR] {msg}\n")

def log_info(msg: str):
    sys.stdout.write(f"[HOOK INFO] {msg}\n")

def get_target_file() -> str | None:
    """Trích xuất file_path từ đối số dòng lệnh hoặc stdin JSON."""
    # 1. Thử đọc từ tham số dòng lệnh trước (tránh block khi test thủ công)
    if len(sys.argv) > 1 and sys.argv[1]:
        return os.path.abspath(sys.argv[1])

    # 2. Thử đọc từ stdin (chuẩn Claude Code PostToolUse Hook)
    try:
        if not sys.stdin.isatty():
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                payload = json.loads(stdin_data)
                tool_input = payload.get("tool_input", {})
                file_path = tool_input.get("file_path") or tool_input.get("path")
                if file_path:
                    return os.path.abspath(file_path)
    except Exception:
        pass

    return None

def check_python_syntax(file_path: str) -> bool:
    """Kiểm tra cú pháp Python. Trả về False nếu có SyntaxError."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        ast.parse(code, filename=file_path)
        return True
    except SyntaxError as e:
        log_err(f"Lỗi cú pháp (SyntaxError) trong {file_path} tại dòng {e.lineno}:{e.offset}: {e.msg}")
        if e.text:
            log_err(f"  -> {e.text.strip()}")
        return False
    except Exception as e:
        log_err(f"Không thể đọc hoặc phân tích cú pháp {file_path}: {e}")
        return False

def check_project_invariants(file_path: str):
    """Cảnh báo nếu phát hiện anti-patterns vi phạm 9 nguyên tắc cốt lõi."""
    if "format_and_lint.py" in file_path or "hooks" in file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Cảnh báo dùng CLAP làm copyright classifier
        if "CLAP" in content and ("copyright" in content.lower() or "license" in content.lower()):
            if "classifier" in content.lower():
                log_err("CẢNH BÁO VI PHẠM NGUYÊN TẮC: Không được dùng CLAP làm copyright classifier!")

        # Cảnh báo except: pass gây nuốt lỗi
        if "except:" in content and "pass" in content:
            # Kiểm tra xem có khối bare except: pass không
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if line.strip() == "except:":
                    if idx + 1 < len(lines) and lines[idx + 1].strip() == "pass":
                        log_err(f"CẢNH BÁO CODE SMELL: Phát hiện 'except: pass' tại dòng {idx + 1}. Bắt buộc xử lý lỗi hoặc log rõ ràng.")
    except Exception:
        pass

def format_python_code(file_path: str):
    """Chạy ruff check --fix và ruff format nếu ruff có sẵn trong môi trường."""
    python_exe = sys.executable
    
    # 1. Thử chạy ruff check --fix
    try:
        subprocess.run(
            [python_exe, "-m", "ruff", "check", "--fix", file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False
        )
    except Exception:
        pass

    # 2. Thử chạy ruff format
    try:
        subprocess.run(
            [python_exe, "-m", "ruff", "format", file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False
        )
    except Exception:
        pass

def check_json_syntax(file_path: str) -> bool:
    """Kiểm tra tính hợp lệ của file JSON."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        log_err(f"Lỗi cú pháp JSON trong {file_path} tại dòng {e.lineno}, cột {e.colno}: {e.msg}")
        return False
    except Exception as e:
        log_err(f"Lỗi đọc file JSON {file_path}: {e}")
        return False

def check_yaml_syntax(file_path: str) -> bool:
    """Kiểm tra tính hợp lệ của file YAML nếu có thư viện PyYAML."""
    try:
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
        return True
    except ImportError:
        # Nếu chưa cài PyYAML thì bỏ qua
        return True
    except Exception as e:
        log_err(f"Lỗi cú pháp YAML trong {file_path}: {e}")
        return False

def check_rules_and_skills_sync(file_path: str):
    """Cảnh báo nhắc nhở Claude Code tự động đồng bộ Rules, Skills và Memory khi code thay đổi."""
    norm_path = file_path.replace("\\", "/").lower()

    # 1. Thay đổi Database Schema
    if any(k in norm_path for k in ["backend/database", "alembic", "database/"]):
        log_info("[SYNC REMINDER] Phát hiện thay đổi Database -> Hãy đồng bộ lại .claude/rules/04-database-schemas.md và memory.md!")

    # 2. Thay đổi API Endpoints / Schemas
    elif any(k in norm_path for k in ["backend/api", "schemas/", "main.py"]):
        log_info("[SYNC REMINDER] Phát hiện thay đổi API/Schemas -> Hãy đồng bộ lại .claude/rules/05-backend-api.md và memory.md!")

    # 3. Thay đổi Rule Engine / Copyright Logic
    elif any(k in norm_path for k in ["configs/rules", "decision_service", "rights_service"]):
        log_info("[SYNC REMINDER] Phát hiện thay đổi Rule Engine -> Hãy đồng bộ lại .claude/rules/03-rule-engine-copyright.md và .claude/skills/rule-engine-verifier/SKILL.md!")

    # 4. Thay đổi Audio Pipeline / AI Models
    elif any(k in norm_path for k in ["audio_service", "fingerprint", "embedding", "mert", "retrieval", "models/"]):
        log_info("[SYNC REMINDER] Phát hiện thay đổi Audio/AI Pipeline -> Hãy đồng bộ lại .claude/rules/02-pipeline-architecture.md và .claude/skills/music-rights-pipeline/SKILL.md!")

    # 5. Thay đổi Thí nghiệm
    elif "experiments/" in norm_path:
        log_info("[SYNC REMINDER] Phát hiện thay đổi Thực nghiệm -> Hãy cập nhật kết quả vào .claude/rules/06-experiments-evaluation.md và .claude/skills/experiment-benchmark/SKILL.md!")

def main():
    target_file = get_target_file()
    if not target_file:
        sys.exit(0)

    if not os.path.exists(target_file):
        sys.exit(0)

    ext = os.path.splitext(target_file)[1].lower()

    # 1. Xử lý file Python (.py)
    if ext == ".py":
        if not check_python_syntax(target_file):
            sys.exit(2)

        format_python_code(target_file)
        check_project_invariants(target_file)
        check_rules_and_skills_sync(target_file)

    # 2. Xử lý file JSON (.json)
    elif ext == ".json":
        if not check_json_syntax(target_file):
            sys.exit(2)
        check_rules_and_skills_sync(target_file)

    # 3. Xử lý file YAML (.yaml, .yml)
    elif ext in [".yaml", ".yml"]:
        if not check_yaml_syntax(target_file):
            sys.exit(2)
        check_rules_and_skills_sync(target_file)

    sys.exit(0)

if __name__ == "__main__":
    main()
