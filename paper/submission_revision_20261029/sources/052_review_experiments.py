"""Exploratory follow-up: fixed folds, ordinal objective, honest baselines, LP proof."""
from pathlib import Path
import argparse
import json
import os
import sys
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from src.data import find_data, load_data, CHEMICAL_DEFECTS
from src.common import save_csv, write_json, sha256, environment
from src.stage3_prediction import tree_model
from src.stage4_models import ordinal_summary
from src.review_diagnostics import distribution_scores, cdf_to_probabilities, conditional_paired_interval, mixture_certificate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    args = parser.parse_args()
    out = ROOT / 'outputs/review_20261028'; out.mkdir(parents=True, exist_ok=True)
    old = ROOT / 'outputs/stage4'
    cfg = json.loads((ROOT / 'config_stage4.json').read_text(encoding='utf-8'))
    data = load_data(find_data(args.data_dir))
    beers = data.meta.beer.to_numpy(); styles = data.meta.tasting_category_fine.to_numpy()
    targets = list(data.Y); X = data.X.to_numpy()
    Y = pd.read_csv(old / 'audit/observed_panel_means_0_6.csv').set_index('beer').loc[beers, targets].to_numpy()
    index = pd.MultiIndex.from_product([beers, targets], names=['beer', 'target'])
    table = pd.read_csv(old / 'ordinal/oof_rating_distributions.csv').set_index(['beer', 'target']).loc[index]
    q = table[[f'observed_panel_p{k}' for k in range(7)]].to_numpy().reshape(250, 50, 7)
    q /= q.sum(-1, keepdims=True)
    probability = {'frequency_ET': table[[f'predicted_p{k}' for k in range(7)]].to_numpy().reshape(250, 50, 7)}
    probability['frequency_ET'] /= probability['frequency_ET'].sum(-1, keepdims=True)
    sums = {m: np.zeros_like(q) for m in ['CDF_ET', 'population', 'style_only']}
    points = {m: np.zeros_like(Y) for m in ['training_median', 'style_median']}
    counts = np.zeros(len(beers)); fold_rows = []; role_rows = []
    scopes = {'core30': np.array([j for j, t in enumerate(targets) if t.startswith(('A_', 'F_'))]), 'all50': np.arange(50)}
    previous_roles = pd.read_csv(old / 'prediction/outer_roles.csv')
    for fold, (train, test) in enumerate(RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=cfg['seed']).split(X, styles)):
        assert set(beers[test]) == set(previous_roles.loc[previous_roles.fold == fold, 'beer'])
        print(f'CDF and baseline comparison {fold + 1}/15', flush=True)
        model = tree_model(cfg).fit(X[train], q[train].cumsum(-1)[..., :-1].reshape(len(train), -1))
        cdf = model.predict(X[test]).reshape(len(test), 50, 6)
        predicted = {'CDF_ET': cdf_to_probabilities(cdf), 'population': np.broadcast_to(q[train].mean(0), (len(test), 50, 7)).copy(), 'style_only': np.empty((len(test), 50, 7))}
        median = np.broadcast_to(np.median(Y[train], axis=0), (len(test), 50)).copy()
        style_median = np.empty_like(median)
        for k, i in enumerate(test):
            group = train[styles[train] == styles[i]]
            if not len(group): group = train
            predicted['style_only'][k] = q[group].mean(0)
            style_median[k] = np.median(Y[group], axis=0)
        points['training_median'][test] += median; points['style_median'][test] += style_median
        for name, p in predicted.items():
            sums[name][test] += p
            for scope, selected in scopes.items():
                metrics = distribution_scores(q[test][:, selected], p[:, selected])
                fold_rows.append({'fold': fold, 'model': name, 'scope': scope, 'n_test': len(test), **{k: float(v.mean()) for k, v in metrics.items()}})
        counts[test] += 1
        role_rows += [{'fold': fold, 'beer': int(beers[i]), 'role': 'test'} for i in test]
    assert np.all(counts == 5)
    for name, p in sums.items(): probability[name] = p / counts[:, None, None]
    summary = []; by_target = []; paired = []; calibration = []
    for name, p in probability.items():
        metric = distribution_scores(q, p)
        grades = p.argmax(-1)
        for scope, selected in scopes.items():
            row = ordinal_summary(q[:, selected], p[:, selected], grades[:, selected])
            # Recompute from the common scoring function for each requested scope.
            for name in ('cdf_distance', 'panel_disagreement', 'expected_individual_RPS', 'brier_distance', 'expected_individual_Brier'):
                row.pop(name, None)
            summary.append({'model': name, 'scope': scope, **row, **{k: float(v[:, selected].mean()) for k, v in metric.items()}})
        for j, target in enumerate(targets):
            by_target.append({'model': name, 'target': target, **ordinal_summary(q[:, j:j+1], None, grades[:, j:j+1]), **{k: float(v[:, j].mean()) for k, v in metric.items()}})
        for scope, selected in scopes.items():
            x = p[:, selected, 0].ravel(); y = q[:, selected, 0].ravel()
            for b in range(10):
                mask = np.minimum((x * 10).astype(int), 9) == b
                if mask.any(): calibration.append({'model': name, 'scope': scope, 'zero_probability_bin': b, 'beer_attribute_pairs': int(mask.sum()), 'predicted_p0': float(x[mask].mean()), 'observed_p0': float(y[mask].mean())})
    for a, b in [('CDF_ET', 'frequency_ET'), ('frequency_ET', 'population'), ('CDF_ET', 'style_only')]:
        for scope, selected in scopes.items():
            d = (distribution_scores(q, probability[a])['cdf_distance'][:, selected] - distribution_scores(q, probability[b])['cdf_distance'][:, selected]).mean(1)
            paired.append({'endpoint': 'cdf_distance', 'scope': scope, 'A': a, 'B': b, **conditional_paired_interval(d, styles)})
    point_preds = {}
    for name in ['mean', 'ridge', 'extra_trees', 'gbr', 'fixed_half_blend', 'concentration_ET']:
        point_preds[name] = pd.read_csv(old / f'prediction/oof_{name}_0_6.csv').set_index('beer').loc[beers, targets].to_numpy()
    for name, v in points.items(): point_preds[name] = v / counts[:, None]
    point_rows = []; style_rows = []; graph_rows = []
    for name, p in point_preds.items():
        for scope, selected in scopes.items():
            error = np.abs(Y[:, selected] - p[:, selected]).mean(1)
            point_rows.append({'model': name, 'scope': scope, 'mae': float(error.mean())})
            for label in np.unique(styles):
                mask = styles == label
                style_rows.append({'model': name, 'scope': scope, 'style': label, 'unique_beers': int(mask.sum()), 'mae': float(error[mask].mean())})
    for name in ['mean', 'training_median', 'extra_trees', 'gbr', 'concentration_ET', 'style_median']:
        for scope, selected in scopes.items():
            d = (np.abs(Y - point_preds['fixed_half_blend'])[:, selected] - np.abs(Y - point_preds[name])[:, selected]).mean(1)
            paired.append({'endpoint': 'panel_mean_MAE', 'scope': scope, 'A': 'fixed_half_blend', 'B': name, **conditional_paired_interval(d, styles)})
    graph_preds = {kind: pd.read_csv(old / f'raw_graph/q2/oof_graph_{kind}.csv').set_index('beer').loc[beers, targets].to_numpy() for kind in ['real', 'none']}
    for scope, selected in scopes.items():
        for kind, p in graph_preds.items(): graph_rows.append({'graph': kind, 'scope': scope, 'aggregated_oof_mae': float(np.abs(Y[:, selected] - p[:, selected]).mean())})
        d = (np.abs(Y - graph_preds['real'])[:, selected] - np.abs(Y - graph_preds['none'])[:, selected]).mean(1)
        paired.append({'endpoint': 'graph_panel_mean_MAE', 'scope': scope, 'A': 'real', 'B': 'none', **conditional_paired_interval(d, styles)})
    roles = pd.read_csv(ROOT / 'outputs/stage3/q3/disjoint_roles.csv')
    concentration = data.calibrated.set_index('beer').loc[beers, CHEMICAL_DEFECTS].to_numpy(float)
    certificates = []; ablations = []
    from scipy.optimize import linprog
    for repeat in range(5):
        role = roles[roles.repeat == repeat].set_index('beer').loc[beers, 'role'].to_numpy()
        anchors = np.flatnonzero((role == 'G_search_training') & (styles == 'Tripel') & np.isfinite(concentration).all(1))
        for i in np.flatnonzero(styles == 'Pils/Lager'):
            d = (concentration[anchors] - concentration[i]).T
            cert = mixture_certificate(d)
            certificates.append({'repeat': repeat, 'beer': int(beers[i]), 'n_anchors': len(anchors), 'certified_zero_only': cert['certified_zero_only'], 'normalized_margin': cert['margin'], 'single_row_blockers': ';'.join(CHEMICAL_DEFECTS[k] for k in cert['single_row_blockers']), **{f'weight_{f}': cert['weights'][k] for k, f in enumerate(CHEMICAL_DEFECTS)}})
            for dropped in [None] + CHEMICAL_DEFECTS:
                keep = [k for k, f in enumerate(CHEMICAL_DEFECTS) if f != dropped]
                a = d[keep]; scales = np.maximum(np.abs(a).max(1), 1e-12)
                solution = linprog(-np.ones(len(anchors)), A_ub=np.vstack([a / scales[:, None], np.ones(len(anchors))]), b_ub=np.r_[np.zeros(len(keep)), .75], bounds=(0, None), method='highs')
                if not solution.success: raise RuntimeError(solution.message)
                ablations.append({'repeat': repeat, 'beer': int(beers[i]), 'dropped_constraint': dropped or 'none', 'max_target_fraction': float(solution.x.sum())})
    for name, frame in [('ordinal_metrics', summary), ('ordinal_by_target', by_target), ('ordinal_fold_metrics', fold_rows), ('zero_calibration', calibration), ('paired_comparisons', paired), ('point_metrics', point_rows), ('point_by_style', style_rows), ('graph_aggregation', graph_rows), ('mixture_certificates', certificates), ('mixture_constraint_ablation', ablations), ('outer_roles', role_rows)]:
        save_csv(pd.DataFrame(frame), out / f'{name}.csv')
    rows = table.reset_index()[['beer', 'target']].copy()
    for k in range(7): rows[f'predicted_p{k}'] = probability['CDF_ET'][..., k].ravel()
    rows['modal_grade'] = probability['CDF_ET'].argmax(-1).ravel()
    rows['median_grade'] = (probability['CDF_ET'].cumsum(-1) >= .5).argmax(-1).ravel()
    save_csv(rows, out / 'cdf_et_oof.csv')
    for name in ['training_median', 'style_median']:
        save_csv(pd.concat([data.meta[['beer']], pd.DataFrame(point_preds[name], columns=targets)], axis=1), out / f'oof_{name}.csv')
    inputs = [old / 'ordinal/oof_rating_distributions.csv', old / 'audit/observed_panel_means_0_6.csv', old / 'prediction/outer_roles.csv', ROOT / 'outputs/stage3/q3/disjoint_roles.csv', ROOT / 'config_stage4.json']
    inputs += list((old / 'prediction').glob('oof_*_0_6.csv')) + list((old / 'raw_graph/q2').glob('oof_graph_*.csv'))
    code = [Path(__file__), ROOT / 'src/review_diagnostics.py', ROOT / 'src/stage3_prediction.py', ROOT / 'src/common.py', ROOT / 'src/stage4_models.py']
    write_json(out / 'run_manifest.json', {**environment(), 'design': 'exploratory_after_stage4_review_fixed_hyperparameters_no_test_tuning', 'config': cfg, 'resamples': 5000, 'unique_beers': 250, 'new_external_samples': 0, 'conditional_intervals': 'fixed_OOF_predictions_stratified_by_style_no_refit_or_selection_uncertainty', 'input_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}, 'code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in code}, 'official_inputs_sha256': {p.name: sha256(p) for p in data.root.iterdir() if p.is_file() and p.suffix in ['.csv', '.xlsx']}, 'result_sha256': {p.name: sha256(p) for p in sorted(out.glob('*.csv'))}})
    print(pd.DataFrame(summary)[['model', 'scope', 'cdf_distance', 'expected_individual_RPS', 'nonzero_panel_mode_accuracy']].to_string(index=False))


if __name__ == '__main__':
    main()
