"""Reconcile newly available raw sensory data, then evaluate absolute scores."""
from pathlib import Path
import argparse
import json
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '2')
from src.common import environment, write_json, sha256
from src.data import load_data, find_data
from src.stage4_audit import reconcile
from src.stage4_models import run_absolute, run_transfer_diagnostics, run_ordinal

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('part', choices=['audit', 'prediction', 'ordinal', 'transfer', 'all'])
parser.add_argument('--external-dir', default=str(ROOT.parent / 'external_sources/zenodo_v4_23624915'))
parser.add_argument('--data-dir')
args = parser.parse_args()
cfg = json.loads((ROOT / 'config_stage4.json').read_text(encoding='utf-8'))
data = load_data(find_data(args.data_dir)); out = ROOT / 'outputs/stage4'
raw, panel, records, concentration, fields = reconcile(data, args.external_dir, out / 'audit', cfg)
if args.part in ['prediction', 'all']:
    run_absolute(data, panel, records, concentration, out, cfg)
if args.part in ['ordinal', 'all']:
    run_ordinal(data, raw, panel, out, cfg, ROOT / 'outputs/stage3')
if args.part in ['transfer', 'all']:
    run_transfer_diagnostics(data, panel, concentration, out, cfg, ROOT / 'outputs/stage3')
write_json(out / f'{args.part}_run_manifest.json', {**environment(), 'config': cfg, 'source_identity_audit': 'audit/audit_summary.json',
                                                'concentration_features': fields, 'part': args.part,
                                                'code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in [ROOT / 'run_stage4.py', ROOT / 'config_stage4.json'] + sorted((ROOT / 'src').glob('*.py'))}})
print(json.dumps({'part': args.part, 'completed': True, 'concentration_fields': len(fields)}))
