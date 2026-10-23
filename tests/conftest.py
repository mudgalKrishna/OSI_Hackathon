import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.audit import run_audit  # noqa: E402
from app.schemas import ExperimentRecord  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def case():
    def _get(case_id: str) -> dict:
        for folder in ("cases", "synthetic"):
            p = ROOT / "data" / folder / f"{case_id}.json"
            if p.exists():
                return load(p)
        raise FileNotFoundError(case_id)
    return _get


@pytest.fixture
def audit():
    return lambda d: run_audit(ExperimentRecord(**d))
