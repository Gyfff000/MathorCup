"""Package only authorized research artifacts; preserve exact bytes for publication."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--payload")
args = parser.parse_args()
root_names = {"README.md", ".gitignore", "config.json", "config_stage2.json", "config_stage3_design.json", "run.py", "run_stage2.py", "requirements.txt", "requirements-lock.txt", "requirements-paper.txt"}
directories = {"src", "tests", "docs", "outputs", "paper", "scripts"}
suffixes = {".md", ".py", ".txt", ".json", ".csv", ".png", ".pdf"}
files = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT)
    if "private" in rel.parts or "__pycache__" in rel.parts:
        continue
    if rel.as_posix() in root_names or (rel.parts[0] in directories and path.suffix.lower() in suffixes):
        assert path.suffix.lower() not in {".xlsx", ".joblib", ".xml"}
        files.append(path)
output = Path(args.out)
output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
    for path in files:
        z.write(path, path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None and len(z.namelist()) == len(files)
if args.payload:
    entries = []
    for path in files:
        raw = path.read_bytes()
        binary = path.suffix.lower() in {".png", ".pdf"}
        entries.append({"path": path.relative_to(ROOT).as_posix(), "encoding": "base64" if binary else "utf-8",
                        "content": base64.b64encode(raw).decode("ascii") if binary else raw.decode("utf-8"),
                        "blob_sha": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()})
    Path(args.payload).write_text(json.dumps(entries, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(json.dumps({"zip": str(output), "files": len(files), "bytes": output.stat().st_size,
                  "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "payload_characters": len(Path(args.payload).read_text(encoding="utf-8")) if args.payload else None}, ensure_ascii=False))
