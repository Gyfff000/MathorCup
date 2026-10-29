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
parser.add_argument('--config', default=str(ROOT / 'config_stage4.json'))
parser.add_argument('--out', default=str(ROOT / 'outputs/stage4'))
parser.add_argument('--stage3', default=str(ROOT / 'outputs/stage3'))
parser.add_argument('--allow-overwrite', action='store_true', help='Permit writing into an existing non-empty output directory')
args = parser.parse_args()
config_path = Path(args.config).resolve()
cfg = json.loads(config_path.read_text(encoding='utf-8'))
if cfg['outer_folds'] < 2 or cfg['outer_repeats'] < 1 or not 0 < cfg['conformal_alpha'] < 1:
    parser.error('outer_folds >= 2, outer_repeats >= 1, and 0 < conformal_alpha < 1 are required')
out = Path(args.out).resolve()
if out.exists() and any(out.iterdir()) and not args.allow_overwrite:
    parser.error('Output directory is not empty. Select a new --out or explicitly use --allow-overwrite.')
data = load_data(find_data(args.data_dir))
raw, panel, records, concentration, fields = reconcile(data, args.external_dir, out / 'audit', cfg)
if args.part in ['prediction', 'all']:
    run_absolute(data, panel, records, concentration, out, cfg)
if args.part in ['ordinal', 'all']:
    run_ordinal(data, raw, panel, out, cfg, Path(args.stage3))
if args.part in ['transfer', 'all']:
    run_transfer_diagnostics(data, panel, concentration, out, cfg, Path(args.stage3))
write_json(out / f'{args.part}_run_manifest.json', {**environment(), 'config': cfg, 'source_identity_audit': 'audit/audit_summary.json',
                                                'concentration_features': fields, 'part': args.part,
                                                'config_source': str(config_path), 'config_sha256': sha256(config_path),
                                                'code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in [ROOT / 'run_stage4.py'] + sorted((ROOT / 'src').glob('*.py'))}})
print(json.dumps({'part': args.part, 'completed': True, 'concentration_fields': len(fields)}))
