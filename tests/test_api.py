from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_and_index():
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/")
    assert r.status_code == 200 and "NitroTrace" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_cases_and_audit_roundtrip():
    cases = client.get("/cases").json()
    ids = [c["case_id"] for c in cases]
    assert ids[:3] == ["synthetic_gas_budget_01", "synthetic_gas_budget_02", "synthetic_gas_budget_03"] or \
        "synthetic_gas_budget_01" in ids
    assert sum(c["group"] == "published" for c in cases) >= 2
    rec = client.get("/cases/synthetic_gas_budget_01").json()
    res = client.post("/audit", json=rec).json()
    assert res["status"] == "contamination_not_excluded"
    assert abs(res["budget_valid_nmol"]["upper"] - 245.24) < 0.01


def test_reports_download():
    rec = client.get("/cases/bragulla2026_pristine_gde_ocv").json()
    for fmt, ctype in [("json", "application/json"), ("csv", "text/csv"), ("md", "text/markdown")]:
        r = client.post(f"/report/{fmt}", json=rec)
        assert r.status_code == 200 and r.headers["content-type"].startswith(ctype)
        assert "attachment" in r.headers["content-disposition"]
    assert client.post("/report/xls", json=rec).status_code == 400


def test_invalid_record_rejected():
    assert client.post("/audit", json={"case_id": "x"}).status_code == 422
    assert client.get("/cases/nope").status_code == 404


def test_sources_and_benchmark():
    s = client.get("/sources").json()
    assert "S2" in s["sources"] and len(s["rules"]) >= 10
    r = client.get("/benchmark")
    assert r.status_code in (200, 404)
