"""Train final 0-6 models or predict from a label-free chemical CSV."""
import argparse
import json
from pathlib import Path
import joblib
from src.common import save_csv, write_json
from src.deployment import fit_final, load_trusted_model, predict

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command', required=True)
train = sub.add_parser('train')
train.add_argument('--chemical', required=True, help='Official attachment 1.csv')
train.add_argument('--means', default=str(ROOT / 'outputs/stage4/audit/observed_panel_means_0_6.csv'))
train.add_argument('--distributions', default=str(ROOT / 'outputs/stage4/ordinal/oof_rating_distributions.csv'))
train.add_argument('--config', default=str(ROOT / 'config_stage4.json'))
train.add_argument('--unit-map', default=str(ROOT / 'outputs/stage4/audit/concentration_feature_registry.csv'))
train.add_argument('--scale-map', default=str(ROOT / 'outputs/stage4/audit/chemical_scale_reconciliation.csv'))
train.add_argument('--model', required=True, help='New output .joblib path')
train.add_argument('--overwrite', action='store_true')
infer = sub.add_parser('predict')
infer.add_argument('--model', required=True, help='Trusted joblib model from this project')
infer.add_argument('--input', required=True, help='Chemical CSV, no labels needed')
infer.add_argument('--out', required=True, help='New output directory')
args = parser.parse_args()
if args.command == 'train':
    path = Path(args.model)
    if (path.exists() or path.with_suffix('.manifest.json').exists()) and not args.overwrite:
        parser.error('Model exists; choose another path or use --overwrite')
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    bundle = fit_final(args.chemical, args.means, args.distributions, cfg, args.unit_map, args.scale_map)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    write_json(path.with_suffix('.manifest.json'), bundle['manifest'])
    print(json.dumps({'model': str(path), 'training_beers': bundle['manifest']['training_beers'],
                      'features': len(bundle['manifest']['features']), 'targets': len(bundle['manifest']['targets'])}))
else:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is not empty; choose a new --out')
    tables = predict(load_trusted_model(args.model), args.input)
    for name, table in tables.items(): save_csv(table, out / f'{name}.csv')
    print(json.dumps({'output': str(out), 'samples': len(tables['panel_mean']), 'targets': len(tables['panel_mean'].columns)-1}))
