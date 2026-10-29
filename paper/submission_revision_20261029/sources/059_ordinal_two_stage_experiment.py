"""Zero/nonzero and conditional-grade ET pilot on fixed, beer-held-out folds."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import preprocessing, save_csv, sha256, write_json
from src.data import META
from src.review_diagnostics import distribution_scores, conditional_paired_interval
from src.stage4_models import ordinal_summary

parser = argparse.ArgumentParser()
parser.add_argument('--chemical', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--trees', type=int, default=32)
args = parser.parse_args()
out = Path(args.out)
if out.exists() and any(out.iterdir()): parser.error('Use a new output directory')
if args.trees < 1: parser.error('--trees must be positive')
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((ROOT / 'config_stage4.json').read_text(encoding='utf8'))
chemical = pd.read_csv(args.chemical)
beers = chemical.beer.to_numpy(); styles = chemical.tasting_category_fine.to_numpy()
X = chemical.drop(columns=META).to_numpy(dtype=float)
base_path = ROOT / 'outputs/stage4/ordinal/oof_rating_distributions.csv'
source = pd.read_csv(base_path)
targets = list(dict.fromkeys(source.target))
idx = pd.MultiIndex.from_product([beers, targets], names=['beer', 'target'])
source = source.set_index(['beer', 'target']).loc[idx]
q = source[[f'observed_panel_p{k}' for k in range(7)]].to_numpy().reshape(len(X), len(targets), 7)
q /= q.sum(-1, keepdims=True)
baseline = source[[f'predicted_p{k}' for k in range(7)]].to_numpy().reshape(q.shape)
baseline /= baseline.sum(-1, keepdims=True)
sum_p = np.zeros_like(q); matched_sum = np.zeros_like(q); counts = np.zeros(len(X)); roles = pd.read_csv(ROOT / 'outputs/stage4/prediction/outer_roles.csv')
start = time.perf_counter()
split = RepeatedStratifiedKFold(n_splits=cfg['outer_folds'], n_repeats=cfg['outer_repeats'], random_state=cfg['seed'])
for fold, (train, test) in enumerate(split.split(X, styles)):
    if set(beers[test]) != set(roles.loc[roles.fold == fold, 'beer']): raise ValueError('Outer roles do not match saved baseline')
    prep = preprocessing().fit(X[train]); a = prep.transform(X[train]); b = prep.transform(X[test])
    matched = ExtraTreesRegressor(n_estimators=args.trees, min_samples_leaf=3, max_features=.8,
                                  random_state=cfg['seed'], n_jobs=2).fit(a, q[train].reshape(len(train), -1)).predict(b)
    matched_sum[test] += matched.reshape(len(test), len(targets), 7)
    zero = ExtraTreesRegressor(n_estimators=args.trees, min_samples_leaf=3, max_features=.8,
                               random_state=cfg['seed'] + fold, n_jobs=2).fit(a, q[train, :, 0]).predict(b)
    zero = np.asarray(zero).reshape(len(test), len(targets)).clip(0, 1)
    prediction = np.zeros((len(test), len(targets), 7)); prediction[:, :, 0] = zero
    for j in range(len(targets)):
        mass = 1 - q[train, j, 0]
        use = mass > 1e-10
        if use.sum() < 3:
            conditional = q[train, j, 1:].sum(0)
            conditional = conditional / conditional.sum() if conditional.sum() > 0 else np.full(6, 1 / 6)
            r = np.broadcast_to(conditional, (len(test), 6))
        else:
            y = q[train[use], j, 1:] / mass[use, None]
            model = ExtraTreesRegressor(n_estimators=args.trees, min_samples_leaf=3, max_features=.8,
                                        random_state=cfg['seed'] + 100 * fold + j, n_jobs=2)
            r = model.fit(a[use], y, sample_weight=mass[use]).predict(b)
            r = np.maximum(r, 0); r /= r.sum(-1, keepdims=True)
        prediction[:, j, 1:] = (1 - zero[:, j, None]) * r
    sum_p[test] += prediction; counts[test] += 1
    print(f'fold {fold+1}/{cfg["outer_folds"]*cfg["outer_repeats"]}; elapsed {time.perf_counter()-start:.1f}s', flush=True)
if not np.all(counts == cfg['outer_repeats']): raise ValueError('Unexpected OOF test counts')
p = sum_p / counts[:, None, None]
matched = matched_sum / counts[:, None, None]
matched /= matched.sum(-1, keepdims=True)
scopes = {'core30': [j for j, t in enumerate(targets) if t.startswith(('A_', 'F_'))], 'all50': list(range(len(targets)))}
summaries = []; comparisons = []
for scope, columns in scopes.items():
    for name, prob in [(f'frequency_ET_{cfg["extra_trees"]}_trees', baseline), (f'frequency_ET_{args.trees}_trees', matched), (f'two_stage_ET_{args.trees}_trees', p)]:
        summaries.append({'scope': scope, 'model': name, **ordinal_summary(q[:, columns], prob[:, columns], prob[:, columns].argmax(-1))})
    for name, comparator in [('frequency_200', baseline), ('frequency_matched', matched)]:
        d = (distribution_scores(q, p)['cdf_distance'][:, columns] - distribution_scores(q, comparator)['cdf_distance'][:, columns]).mean(1)
        comparisons.append({'scope': scope, 'comparison': f'two_stage_minus_{name}', **conditional_paired_interval(d, styles)})
rows = pd.DataFrame({'beer': np.repeat(beers, len(targets)), 'target': np.tile(targets, len(beers))})
for k in range(7): rows[f'predicted_p{k}'] = p[:, :, k].ravel()
rows['predicted_modal_grade'] = p.argmax(-1).ravel()
save_csv(rows, out / 'oof_distribution.csv'); save_csv(pd.DataFrame(summaries), out / 'metrics.csv'); save_csv(pd.DataFrame(comparisons), out / 'paired_comparison.csv')
matched_rows = pd.DataFrame({'beer': np.repeat(beers, len(targets)), 'target': np.tile(targets, len(beers)), 'predicted_modal_grade': matched.argmax(-1).ravel()})
for k in range(7): matched_rows[f'predicted_p{k}'] = matched[:, :, k].ravel()
save_csv(matched_rows, out / 'oof_frequency_matched.csv')
write_json(out / 'manifest.json', {'trees_per_two_stage_model': args.trees, 'baseline_trees': cfg['extra_trees'],
    'folds': cfg['outer_folds'], 'repeats': cfg['outer_repeats'], 'seed': cfg['seed'],
    'design': 'exploratory; conditional nonzero distributions weighted by training panel nonzero mass; fixed saved folds; no calibration',
    'conditional_interval': 'beer stratified bootstrap of fixed OOF predictions; no refitting uncertainty',
    'source_sha256': {'baseline_ratings': sha256(base_path), 'chemical_file': sha256(args.chemical)}})
print(pd.DataFrame(summaries)[['scope','model','cdf_distance','expected_individual_RPS','nonzero_panel_mode_accuracy']].to_string(index=False))
