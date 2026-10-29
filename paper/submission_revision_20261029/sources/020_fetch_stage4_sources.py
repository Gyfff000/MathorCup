"""Fetch two public author sources with pinned MD5; keep raw files outside repo."""
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--out', default=str(ROOT.parent / 'external_sources/zenodo_v4_23624915'))
args = parser.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((ROOT / 'config_stage4.json').read_text(encoding='utf-8'))
for filename, key in [('Raw sensory data anonymized.csv', 'raw_md5'), ('Supplemental Files and Figure source files.xlsx', 'workbook_md5')]:
    path = out / filename
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == cfg[key]:
        print('Verified existing source: ' + filename); continue
    url = cfg['external_record'] + '/files/' + quote(filename) + '?download=1'
    raw = urlopen(url, timeout=60).read()
    if hashlib.md5(raw).hexdigest() != cfg[key]:
        raise ValueError('Publisher checksum mismatch: ' + filename)
    path.write_bytes(raw)
    print('Downloaded and verified: ' + filename)
