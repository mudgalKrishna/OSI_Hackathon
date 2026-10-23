import pytest

from app import extraction

pytestmark = pytest.mark.skipif(not extraction.available(), reason="PyMuPDF not installed")

TEXT = ("Methods. The N2 gas was purged at a flow rate of 20 mL min-1 through the cell for 2 h. "
        "The electrolyte (35 mL, 0.1 M KOH) was analysed for NH3 by the indophenol method. "
        "An Ar control at open circuit gave a background of 0.8 µg L-1 NH4+. "
        "NOx impurities in the N2 supply were below 5 ppb. 15N2 isotope experiments were not performed. ") * 3


def _pdf() -> bytes:
    doc = extraction.pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(extraction.pymupdf.Rect(40, 40, 560, 800), TEXT, fontsize=9)
    return doc.tobytes()


def test_passages_and_regex_candidates():
    out = extraction.extract(_pdf())
    assert out["n_pages"] == 1
    topics = {t for p in out["passages"] for t in p["topics"]}
    assert {"gas_flow", "impurity", "blank"} <= topics
    vals = {k for p in out["passages"] for k in p["regex_values"]}
    assert "flow" in vals and "gas_fraction" in vals
    assert "llm" not in out  # no model call unless requested


def test_rejects_non_pdf():
    with pytest.raises(extraction.ExtractionError):
        extraction.extract(b"not a pdf")
