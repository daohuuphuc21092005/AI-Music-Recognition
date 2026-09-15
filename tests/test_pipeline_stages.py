"""Mọi bước mà pipeline báo ra phải có trong PipelineStage, nếu không GET /jobs trả 500."""
import re
from pathlib import Path

from backend.schemas.analysis import PipelineStage

SERVICES = Path(__file__).resolve().parent.parent / "backend" / "services"
REPORTED = re.compile(r"""report\(\s*["']([A-Z_]+)["']\s*\)""")


def test_moi_buoc_pipeline_bao_ra_deu_khai_bao_trong_enum():
    reported = set()
    for name in ("cascade_service.py", "analysis_pipeline.py"):
        reported |= set(REPORTED.findall((SERVICES / name).read_text(encoding="utf-8")))

    assert reported, "không tìm thấy lời gọi report(...) nào — regex đã lệch với code"
    missing = reported - {stage.value for stage in PipelineStage}
    assert not missing, f"bước chưa khai báo trong PipelineStage: {sorted(missing)}"
