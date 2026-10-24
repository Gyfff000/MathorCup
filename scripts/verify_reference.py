"""Validate the actual paper and saved evidence, without publishing host identifiers."""
from pathlib import Path
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage2"
reader = PdfReader(ROOT / "paper/论文参考稿.pdf")
texts = [page.extract_text() for page in reader.pages]
assert len(reader.pages) == 13
assert "摘 要" in texts[0] and "目 录" in texts[1]
assert "1 问题重述与研究范围" in texts[2]
assert all("\x00" not in text for text in texts)
assert reader.metadata.get("/Author", "") == ""
assert all("Gyfff000" not in text and "github.com" not in text for text in texts)
assert all(abs(float(p.mediabox.width) - 595.2756) < 0.1 for p in reader.pages)
content = json.loads((ROOT / "paper/paper_content.json").read_text(encoding="utf-8"))
for block in content["blocks"]:
    if block["type"] == "heading" and block["level"] == 1:
        title = block["text"]
        expected = next(i-1 for i, page in enumerate(texts[2:], 2) if title in page)
        destination = next(item for item in reader.outline if not isinstance(item, list) and item.title == title)
        assert reader.get_destination_page_number(destination)-1 == expected
with (OUT / "q1/metrics_summary.csv").open(encoding="utf-8-sig", newline="") as f:
    metrics = list(csv.DictReader(f))
gbr = next(row for row in metrics if row["model"] == "gbr_fixed")
assert abs(float(gbr["mae"]) - 0.339759) < 0.000001
suite = ET.parse(OUT / "tests.xml").getroot().find("testsuite")
assert int(suite.attrib["tests"]) == 23
assert all(int(suite.attrib[k]) == 0 for k in ["errors", "failures", "skipped"])
summary = {
    "date": "2026-10-24",
    "tests": {k: int(suite.attrib[k]) for k in ["tests", "errors", "failures", "skipped"]},
    "test_names": [item.attrib["name"] for item in suite.findall("testcase")],
    "paper": {"physical_pages": len(reader.pages), "abstract_page": 1, "contents_page": 2,
              "body_starts_physical_page": 3, "body_starts_printed_page": 1,
              "equations": sum(b["type"] == "equation" for b in content["blocks"]),
              "figures": sum(b["type"] == "figure" for b in content["blocks"]),
              "missing_glyph_nulls": 0, "author_metadata_empty": True,
              "visual_review": "all 13 rendered pages inspected; final amendment on defect names reviewed",
              "sha256": hashlib.sha256((ROOT / "paper/论文参考稿.pdf").read_bytes()).hexdigest()},
    "scientific_scope": "Exploratory AI-assisted reference; no absolute-score recovery, causal or tasting validation claimed."
}
(OUT / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
files = [p for p in OUT.rglob("*") if p.is_file() and p.name not in ["tests.xml", "result_checksums.json"] and "private" not in p.parts]
checksums = [{"file": p.relative_to(OUT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
(OUT / "result_checksums.json").write_text(json.dumps(checksums, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"pages": len(reader.pages), "tests_passed": 23, "hashed_results": len(checksums)}, ensure_ascii=False))
