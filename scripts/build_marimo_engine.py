"""Generate the marimo notebook's engine cell from the app/ package (single source of truth).

The notebook has to run where the package is not installed (e.g. a molab sandbox), so the engine
modules are concatenated into one cell: relative imports are dropped, module-level private names are
renamed (marimo keeps `_names` local to their cell), and file-based data loading is replaced by the
embedded DATA dict from the notebook's data cell.

Run:  python scripts/build_marimo_engine.py > engine_cell.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
MODULES = ["schemas", "units", "budget", "coverage", "controls", "audit", "reports",
           "inversion", "planner", "whatif", "validation"]
DROP_IMPORTS = re.compile(r"^(?:from __future__ |from \.|from pydantic |from typing |from functools |from pathlib |"
                          r"import (?:json|csv|io|math|copy|time)\b).*\n", re.M)


def port(mod: str) -> str:
    s = (APP / f"{mod}.py").read_text(encoding="utf-8")
    s = re.sub(r'^"""(.*?)"""\n', lambda m: "# " + m.group(1).strip().replace("\n", "\n# ") + "\n", s, count=1, flags=re.S)
    s = re.sub(r"^from \.\w+ import \([^)]*\)\n", "", s, flags=re.M)
    s = DROP_IMPORTS.sub("", s)
    s = re.sub(r"\bbudget\.(?=[a-z_]+\()", "", s)
    s = re.sub(r"\bcoverage\.(resolve|complete)\(", r"coverage_\1(", s)
    s = re.sub(r"\bcontrols\.recommend\(", "recommend(", s)
    if mod == "coverage":
        s = s.replace("def resolve(", "def coverage_resolve(").replace("def complete(", "def coverage_complete(")
    if mod == "controls":
        s = re.sub(r"^# Repository checkout.*\n^_PKG = .*\n", "", s, flags=re.M)
        s = re.sub(r"^DATA_DIR = .*\n", "DATA_DIR = None  # notebook: records come from the embedded DATA\n", s, flags=re.M)
        s = re.sub(r"RULES_PATH = .*\nSOURCES_PATH = .*\n", "", s)
        s = s.replace("@lru_cache\n", "")
        s = s.replace('json.loads(RULES_PATH.read_text(encoding="utf-8"))', 'DATA["protocols/rules.json"]')
        s = s.replace('json.loads(SOURCES_PATH.read_text(encoding="utf-8"))', 'DATA["protocols/sources.json"]')
    if mod == "validation":
        s = re.sub(r"^DATA_DIR = .*\n", "DATA_DIR = None  # notebook: records come from the embedded DATA\n", s, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", s).strip() + "\n"


def build() -> str:
    body = "".join(f"\n# ======================= {m} =======================\n" + port(m) for m in MODULES)
    private = set(re.findall(r"^def (_[a-zA-Z]\w*)", body, re.M)) | set(re.findall(r"^(_[A-Za-z]\w*) =", body, re.M))
    for name in sorted(private, key=len, reverse=True):
        body = re.sub(rf"\b{name}\b", "nt" + name, body)
    return ("# NitroTrace engine, generated from the app/ package by scripts/build_marimo_engine.py — edit the package,\n"
            "# not this cell. All amounts are nmol of ammonia-equivalent N; missing evidence is unknown, never zero.\n" + body)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(build())
