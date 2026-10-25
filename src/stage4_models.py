"""Absolute panel-mean prediction, factor checks, and physical-scale diagnostics.
AI-assisted research implementation, 2026-10-25; no original algorithm claim.
"""
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from .common import preprocessing, save_csv, write_json, progress
from .data import NON_MOLECULES, CHEMICAL_DEFECTS, AGGREGATES
from .q1 import metric_rows
from .stage2 import conformal_quantile
from .stage3_prediction import fit_pair, predict_pair, tree_model
from .stage4_audit import panel_rating_distribution


def calibrated_from_training(z_train, raw_train, z_pred):
    return np.column_stack([IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=6).fit(z_train[:, j], raw_train[:, j]).predict(z_pred[:, j])
                            for j in range(raw_train.shape[1])])


def ridge_search(X, Y, core, cfg):
    return GridSearchCV(Pipeline([('prep', preprocessing()), ('model', Ridge())]),
                        {'model__alpha': cfg['ridge_alphas']}, cv=KFold(3, shuffle=True, random_state=cfg['seed']),
                        scoring=lambda est, xx, yy: -np.abs(yy[:, core] - est.predict(xx)[:, core]).mean(), n_jobs=1).fit(X, Y).best_estimator_


def raw_top3(model, fields, targets):
    candidates = np.array([j for j, f in enumerate(fields) if f not in NON_MOLECULES])
    coef = model.named_steps['model'].coef_
    return [(t, int(j), float(coef[k, j]), rank + 1) for k, t in enumerate(targets)
            for rank, j in enumerate(candidates[np.argsort(-np.abs(coef[k, candidates]), kind='stable')[:3]])]


def run_absolute(data, panel, records, concentration, out, cfg):
    dest = Path(out) / 'prediction'; dest.mkdir(parents=True, exist_ok=True)
    X, Y, C = data.X.to_numpy(), panel.to_numpy(), concentration.to_numpy()
    targets, fields = list(data.Y), list(data.X)
    core = np.array([j for j, t in enumerate(targets) if t.startswith(('A_', 'F_'))])
    labels, beers = data.meta.tasting_category_fine.to_numpy(), data.meta.beer.to_numpy()
    names = ['mean', 'ridge', 'extra_trees', 'gbr', 'fixed_half_blend', 'concentration_ET', 'standardized_ET_train_isotonic']
    sums = {m: np.zeros_like(Y) for m in names}; count = np.zeros(len(Y)); folds, factors, sensitivity, roles = [], [], [], []
    start = time.perf_counter()
    for fold, (train, test) in enumerate(RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=cfg['seed']).split(X, labels)):
        progress(f'S4 absolute fold {fold+1}/15; {(time.perf_counter()-start):.0f}s')
        et, gb = fit_pair(X[train], Y[train], cfg)
        pe, pg = predict_pair((et, gb), X[test])
        ridge = ridge_search(X[train], Y[train], core, cfg)
        cp = tree_model(cfg).fit(C[train], Y[train]).predict(C[test])
        old = tree_model(cfg).fit(X[train], data.Y.to_numpy()[train]).predict(X[test])
        iso = calibrated_from_training(data.Y.to_numpy()[train], Y[train], old)
        predictions = {'mean': np.broadcast_to(Y[train].mean(0), Y[test].shape), 'ridge': ridge.predict(X[test]),
                       'extra_trees': pe, 'gbr': pg, 'fixed_half_blend': (pe + pg) / 2, 'concentration_ET': cp,
                       'standardized_ET_train_isotonic': iso}
        for name, pred in predictions.items():
            pred = np.clip(pred, 0, 6); sums[name][test] += pred
            for scope, indices in [('core30', core), ('all50', np.arange(50))]:
                folds.append({'fold': fold, 'repeat': fold // 3, 'model': name, 'scope': scope,
                              'mae_0_6': float(np.abs(Y[test][:, indices] - pred[:, indices]).mean()), 'n_test': len(test)})
        count[test] += 1
        record_model = tree_model(cfg).fit(X[train], records.to_numpy()[train])
        record_pred = record_model.predict(X[test])
        sensitivity.append({'fold': fold, 'person_balanced_ET_mae_own_endpoint': float(np.abs(Y[test] - pe).mean()),
                            'record_weighted_ET_mae_own_endpoint': float(np.abs(records.to_numpy()[test] - record_pred).mean()),
                            'prediction_aggregation_difference': float(np.abs(record_pred - pe).mean())})
        roles.extend({'fold': fold, 'repeat': fold // 3, 'beer': int(beers[i]), 'role': 'outer_test'} for i in test)
        # Linear model permits exact input-coordinate permutation changes without refits.
        zp = ridge.named_steps['prep'].transform(X[test]); rp = ridge.predict(X[test]); rng = np.random.default_rng(cfg['seed'] + fold)
        for target, j, coefficient, rank in raw_top3(ridge, fields, targets):
            k = targets.index(target); baseline = np.abs(Y[test, k] - rp[:, k]).mean(); ordinary, within = [], []
            for _ in range(20):
                permutation = rng.permutation(len(test))
                ordinary.append(np.abs(Y[test, k] - (rp[:, k] + coefficient * (zp[permutation, j] - zp[:, j]))).mean() - baseline)
                permutation = np.arange(len(test))
                for style in np.unique(labels[test]):
                    ids = np.flatnonzero(labels[test] == style); permutation[ids] = rng.permutation(ids)
                within.append(np.abs(Y[test, k] - (rp[:, k] + coefficient * (zp[permutation, j] - zp[:, j]))).mean() - baseline)
            factors.append({'fold': fold, 'target': target, 'field': fields[j], 'rank': rank, 'coefficient': coefficient,
                            'ordinary_delta_mae': np.mean(ordinary), 'within_style_delta_mae': np.mean(within)})
        save_csv(pd.DataFrame(folds), dest / 'fold_metrics.csv')
    metric = []
    for name in names:
        pred = sums[name] / count[:, None]
        save_csv(pd.concat([data.meta[['beer']], pd.DataFrame(pred, columns=targets)], axis=1), dest / f'oof_{name}_0_6.csv')
        metric += metric_rows(Y, pred, targets, name)
    detail = pd.DataFrame(metric); save_csv(detail, dest / 'metrics_by_target.csv')
    summaries = []
    for scope, selected in [('core30', [targets[j] for j in core]), ('all50', targets)]:
        f = detail[detail.target.isin(selected)].groupby('model')[['mae', 'rmse', 'r2']].mean().reset_index(); f.insert(0, 'scope', scope); summaries.append(f)
    save_csv(pd.concat(summaries), dest / 'metrics_by_scope.csv')
    save_csv(pd.DataFrame(roles), dest / 'outer_roles.csv')
    save_csv(pd.DataFrame(sensitivity), dest / 'aggregation_sensitivity.csv')
    effect = pd.DataFrame(factors); save_csv(effect, dest / 'raw_factor_crossfit.csv')
    # Prespecified ET is used for uncertainty; evaluation does not select the model.
    intervals, coverage, interval_roles = [], [], []
    for repeat in range(cfg['conformal_repeats']):
        train, rest = train_test_split(np.arange(len(X)), test_size=.4, random_state=cfg['seed'] + 71 + repeat)
        calibration, test = train_test_split(rest, test_size=.5, random_state=cfg['seed'] + 72 + repeat)
        model = tree_model(cfg).fit(X[train], Y[train]); pred = model.predict(X[test])
        q = conformal_quantile(np.abs(Y[calibration] - model.predict(X[calibration])), .1)
        lo, hi = np.maximum(0, pred - q), np.minimum(6, pred + q)
        for j, target in enumerate(targets):
            coverage.append({'repeat': repeat, 'target': target, 'coverage': float(((Y[test, j] >= lo[:, j]) & (Y[test, j] <= hi[:, j])).mean()),
                             'mean_width_0_6': float((hi[:, j] - lo[:, j]).mean())})
        intervals.extend({'repeat': repeat, 'beer': int(beers[i]), 'target': target, 'prediction_0_6': pred[k, j],
                          'lower_0_6': lo[k, j], 'upper_0_6': hi[k, j], 'observed_panel_mean': Y[i, j]}
                         for k, i in enumerate(test) for j, target in enumerate(targets))
        interval_roles.extend({'repeat': repeat, 'beer': int(beers[i]), 'role': 'fit' if i in train else 'calibration' if i in calibration else 'test'} for i in range(len(X)))
    save_csv(pd.DataFrame(intervals), dest / 'absolute_panel_mean_intervals.csv')
    save_csv(pd.DataFrame(coverage), dest / 'conformal_coverage.csv')
    save_csv(pd.DataFrame(interval_roles), dest / 'conformal_roles.csv')
    final_ridge = ridge_search(X, Y, core, cfg); alpha = float(final_ridge.named_steps['model'].alpha)
    final = raw_top3(final_ridge, fields, targets); frequencies = {}
    rng = np.random.default_rng(cfg['seed'] + 333)
    for _ in range(200):
        ii = rng.integers(len(X), size=len(X))
        m = Pipeline([('prep', preprocessing()), ('model', Ridge(alpha=alpha))]).fit(X[ii], Y[ii])
        for target, j, _, _ in raw_top3(m, fields, targets): frequencies[(target, fields[j])] = frequencies.get((target, fields[j]), 0) + 1
    grouped = effect.groupby(['target', 'field']).agg(selected_folds=('fold', 'size'), ordinary_delta_mae=('ordinary_delta_mae', 'mean'), within_style_delta_mae=('within_style_delta_mae', 'mean'))
    metadata = data.chemical_meta.set_index('field'); top_rows = []
    for target, j, coefficient, rank in final:
        field = fields[j]; key = (target, field); f = frequencies.get(key, 0) / 200
        evidence = grouped.loc[key] if key in grouped.index else pd.Series({'selected_folds': 0, 'ordinary_delta_mae': np.nan, 'within_style_delta_mae': np.nan})
        top_rows.append({'target': target, 'rank': rank, 'field': field, 'coefficient_raw_score_per_standardized_input': coefficient,
                         'bootstrap_frequency_fixed_alpha': f, **evidence.to_dict(),
                         'descriptive_support': bool(f >= .5 and evidence.selected_folds >= 5 and evidence.ordinary_delta_mae > 0 and evidence.within_style_delta_mae > 0),
                         'unit': metadata.at[field, 'unit'], 'origin': metadata.at[field, 'origin_paper'], 'causal_validated': False})
    save_csv(pd.DataFrame(top_rows), dest / 'absolute_raw_top3.csv')
    write_json(dest / 'status.json', {'complete': True, 'elapsed_seconds': time.perf_counter() - start, 'outer_fits': 15,
                                     'selected_full_ridge_alpha': alpha, 'absolute_endpoint': cfg['primary_absolute_target'],
                                     'source_identity_reconstructed': True, 'new_external_beers': 0})


def mix_concentrations(source, anchor, weights):
    return source[None, :] + np.asarray(weights)[:, None] * (anchor[None, :] - source[None, :])


def exact_nonincrease(values, source, positions):
    if not np.isfinite(source[positions]).all():
        return np.zeros(len(values), dtype=bool)
    return np.isfinite(values[:, positions]).all(1) & (values[:, positions] <= source[positions] + 1e-12).all(1)


def ordinal_summary(observed, probabilities, grades):
    axis = np.arange(7)
    individual_mae = (observed * np.abs(axis - grades[..., None])).sum(-1)
    individual_accuracy = np.take_along_axis(observed, grades[..., None], axis=-1)[..., 0]
    modal_accuracy = grades == observed.argmax(-1)
    result = {'expected_individual_grade_MAE': float(individual_mae.mean()),
              'expected_individual_exact_accuracy': float(individual_accuracy.mean()),
              'observed_panel_modal_grade_accuracy': float(modal_accuracy.mean()),
              'predicted_zero_fraction': float((grades == 0).mean()),
              'observed_panel_zero_mode_fraction': float((observed.argmax(-1) == 0).mean())}
    nonzero = observed.argmax(-1) > 0
    result['nonzero_panel_mode_accuracy'] = float(modal_accuracy[nonzero].mean()) if nonzero.any() else np.nan
    if probabilities is not None:
        result.update(RPS=float(((probabilities.cumsum(-1)[..., :-1] - observed.cumsum(-1)[..., :-1]) ** 2).mean()),
                      Brier=float(((probabilities - observed) ** 2).sum(-1).mean()),
                      expected_panel_mean_MAE=float(np.abs((probabilities * axis).sum(-1) - (observed * axis).sum(-1)).mean()))
    return result


def run_ordinal(data, raw, panel, out, cfg, previous):
    dest = Path(out) / 'ordinal'; dest.mkdir(parents=True, exist_ok=True)
    targets = list(data.Y); X = data.X.to_numpy(); Y = panel.to_numpy(); beers = data.meta.beer.to_numpy()
    labels = data.meta.tasting_category_fine.to_numpy(); observed = panel_rating_distribution(raw, targets, beers)
    core = np.array([j for j, t in enumerate(targets) if t.startswith(('A_', 'F_'))]); scopes = [('core30', core), ('all50', np.arange(50))]
    sums = np.zeros_like(observed); base_sum = np.zeros_like(observed); point_sum = np.zeros_like(Y); counts = np.zeros(len(X)); folds, stress = [], []
    for fold, (train, test) in enumerate(RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=cfg['seed']).split(X, labels)):
        progress(f'S4 ordinal distribution fold {fold+1}/15')
        model = tree_model(cfg).fit(X[train], observed[train].reshape(len(train), -1))
        p = model.predict(X[test]).reshape(len(test), 50, 7); p /= p.sum(-1, keepdims=True)
        base = np.broadcast_to(observed[train].mean(0), p.shape)
        point = tree_model(cfg).fit(X[train], Y[train]).predict(X[test])
        sums[test] += p; base_sum[test] += base; point_sum[test] += point; counts[test] += 1
        for name, prob, grades in [('ET_rating_distribution', p, p.argmax(-1)),
                                   ('ET_rating_distribution_median', p, (p.cumsum(-1) >= .5).argmax(-1)),
                                   ('training_population_distribution', base, base.argmax(-1)),
                                   ('ET_mean_round_half_up', None, np.floor(point + .5).clip(0, 6).astype(int))]:
            for scope, selected in scopes:
                folds.append({'fold': fold, 'scope': scope, 'method': name,
                              **ordinal_summary(observed[test][:, selected], None if prob is None else prob[:, selected], grades[:, selected])})
        rng = np.random.default_rng(cfg['seed'] + fold)
        for rate in [0, .05, .1, .2]:
            masked = X[test].copy(); eligible = np.flatnonzero(np.isfinite(masked).ravel()); n = int(rate * len(eligible))
            if n: masked.ravel()[rng.choice(eligible, n, replace=False)] = np.nan
            changed = model.predict(masked).reshape(len(test), 50, 7)
            for scope, selected in scopes:
                stress.append({'fold': fold, 'scope': scope, 'additional_MCAR_rate': rate,
                               'modal_grade_change_fraction': float((changed.argmax(-1)[:, selected] != p.argmax(-1)[:, selected]).mean()),
                               **ordinal_summary(observed[test][:, selected], changed[:, selected], changed.argmax(-1)[:, selected])})
    probability = sums / counts[:, None, None]; baseline = base_sum / counts[:, None, None]; point = point_sum / counts[:, None]
    results = []
    for method, prob, grades in [('ET_rating_distribution', probability, probability.argmax(-1)),
                                ('ET_rating_distribution_median', probability, (probability.cumsum(-1) >= .5).argmax(-1)),
                                ('training_population_distribution', baseline, baseline.argmax(-1)),
                                ('ET_mean_round_half_up', None, np.floor(point + .5).clip(0, 6).astype(int))]:
        for scope, selected in scopes:
            results.append({'scope': scope, 'method': method, **ordinal_summary(observed[:, selected], None if prob is None else prob[:, selected], grades[:, selected])})
    save_csv(pd.DataFrame(results), dest / 'metrics_by_scope.csv')
    save_csv(pd.DataFrame(folds), dest / 'fold_metrics.csv'); save_csv(pd.DataFrame(stress), dest / 'missingness_stress.csv')
    save_csv(pd.concat([data.meta[['beer']], pd.DataFrame(probability.argmax(-1), columns=targets)], axis=1), dest / 'oof_modal_grades_0_6.csv')
    save_csv(pd.concat([data.meta[['beer']], pd.DataFrame((probability.cumsum(-1) >= .5).argmax(-1), columns=targets)], axis=1), dest / 'oof_median_grades_0_6.csv')
    save_csv(pd.concat([data.meta[['beer']], pd.DataFrame(np.floor(point + .5).clip(0, 6).astype(int), columns=targets)], axis=1), dest / 'oof_rounded_panel_mean_grades_0_6.csv')
    rows = []
    for i, beer in enumerate(beers):
        for j, target in enumerate(targets):
            rows.append({'beer': int(beer), 'target': target, 'predicted_modal_grade': int(probability[i, j].argmax()),
                         'observed_panel_modal_grade': int(observed[i, j].argmax()),
                         **{f'predicted_p{k}': probability[i, j, k] for k in range(7)},
                         **{f'observed_panel_p{k}': observed[i, j, k] for k in range(7)}})
    save_csv(pd.DataFrame(rows), dest / 'oof_rating_distributions.csv')
    roles = pd.read_csv(Path(previous) / 'q3/disjoint_roles.csv'); generated = pd.read_csv(Path(previous) / 'q3/generated_profiles_first_split.csv')
    features = [f for f in data.X if f not in AGGREGATES]; repeat0 = roles[roles.repeat == 0].set_index('beer').role.reindex(beers)
    for name, role in [('G', 'G_search_training'), ('J', 'J_evaluator_training')]:
        subset = np.flatnonzero(repeat0.to_numpy() == role)
        model = tree_model(cfg).fit(data.X[features].to_numpy()[subset], observed[subset].reshape(len(subset), -1))
        p = model.predict(generated[features].to_numpy()).reshape(len(generated), 50, 7)
        save_csv(pd.concat([generated[['generated_id', 'method', 'style']], pd.DataFrame(p.argmax(-1), columns=targets)], axis=1),
                 dest / f'generated_modal_grade_proxy_{name}_first_split.csv')
    write_json(dest / 'status.json', {'complete': True, 'beer_grouped_outer_fits': 15, 'grade_definition': 'modal_prediction_of_actual_0_6_panelist_rating_distribution',
                                    'zero_definition': 'most_probable_panelist_rating_zero_not_chemical_absence',
                                    'equal_panelist_and_within_person_repeat_weights': True, 'probabilities_claimed_calibrated': False,
                                    'new_physical_measurements': False})


def run_transfer_diagnostics(data, panel, concentration, out, cfg, previous):
    dest = Path(out) / 'transfer'; dest.mkdir(parents=True, exist_ok=True)
    previous = Path(previous); roles = pd.read_csv(previous / 'q3/disjoint_roles.csv')
    profiles = pd.read_csv(previous / 'q3/transfer_profiles.csv'); old_eval = pd.read_csv(previous / 'q3/transfer_independent_evaluation.csv')
    targets = list(data.Y); stale = targets.index('stale_hops'); beers = data.meta.beer.to_numpy(); labels = data.meta.tasting_category_fine.to_numpy()
    nonaggregate = [f for f in data.X if f not in AGGREGATES]; X = data.X[nonaggregate].to_numpy(); Y = panel.to_numpy(); C = concentration.to_numpy(); features = list(concentration)
    positions = [features.index(f) for f in CHEMICAL_DEFECTS]
    frozen_rows, mixture_rows, mixture_profiles, locks, diagnostics, lp_rows = [], [], [], [], [], []
    for repeat in range(5):
        progress(f'S4 frozen-candidate and concentration mixture evaluation {repeat+1}/5')
        role = roles[roles.repeat == repeat].set_index('beer').role.reindex(beers).to_numpy()
        G = np.flatnonzero(role == 'G_search_training'); J = np.flatnonzero(role == 'J_evaluator_training'); T = np.flatnonzero(role == 'T_real_holdout')
        assert len(G) == len(J) == 100 and len(T) == 50 and not set(G) & set(J)
        # Old candidates are already frozen; no raw-scale J feedback changes them.
        graw = tree_model(cfg).fit(X[G], Y[G]); jraw = tree_model(cfg).fit(X[J], Y[J])
        if repeat == 0:
            generated = pd.read_csv(previous / 'q3/generated_profiles_first_split.csv')
            for name, predictor in [('G', graw), ('J', jraw)]:
                portrait = pd.DataFrame(predictor.predict(generated[nonaggregate].to_numpy()), columns=targets)
                save_csv(pd.concat([generated[['generated_id', 'method', 'style']].reset_index(drop=True), portrait], axis=1),
                         dest / f'generated_raw_score_proxy_{name}_first_split.csv')
        old = profiles[profiles.repeat == repeat].copy(); candidate = old[nonaggregate].to_numpy(); ids = pd.Index(beers).get_indexer(old.beer)
        sp, cp = jraw.predict(X[ids])[:, stale], jraw.predict(candidate)[:, stale]
        ga, gc = graw.predict(X[ids])[:, stale], graw.predict(candidate)[:, stale]
        prior = old.merge(old_eval[['repeat', 'beer', 'radius', 'evaluator_proxy_acceptance']], on=['repeat', 'beer', 'radius'], validate='one_to_one')
        for k, row in old.reset_index(drop=True).iterrows():
            feasible = not row.status.startswith('no_feasible')
            frozen_rows.append({'repeat': repeat, 'beer': int(row.beer), 'radius': row.radius, 'source_role': role[ids[k]],
                                'original_search_feasible': feasible, 'raw_G_stale_source': ga[k], 'raw_G_stale_candidate': gc[k],
                                'raw_J_stale_source': sp[k], 'raw_J_stale_candidate': cp[k],
                                'raw_G_nonincrease': bool(gc[k] <= ga[k] + 1e-9), 'raw_J_nonincrease': bool(cp[k] <= sp[k] + 1e-9),
                                'original_proxy_acceptance': bool(prior.iloc[k].evaluator_proxy_acceptance),
                                'original_and_raw_proxy_support': bool(feasible and prior.iloc[k].evaluator_proxy_acceptance and gc[k] <= ga[k] + 1e-9 and cp[k] <= sp[k] + 1e-9),
                                'physical_validation': False})
        # New mixtures use only documented concentration-unit fields, not pH/pressure/A.U.
        generator_prep = preprocessing().fit(C[G]); fill = generator_prep.named_steps['impute'].statistics_
        source_data = np.where(np.isfinite(C), C, fill)
        search = Pipeline([('prep', preprocessing()), ('model', LogisticRegression(C=1, max_iter=3000, random_state=cfg['seed']))]).fit(C[G], labels[G])
        sensory = tree_model(cfg).fit(C[G], Y[G])
        consensus = Pipeline([('prep', preprocessing()), ('model', Ridge(alpha=1000))]).fit(C[G], Y[G])
        target_position = list(search.classes_).index('Tripel'); anchors = G[labels[G] == 'Tripel']; locked = []
        for i in np.flatnonzero(labels == 'Pils/Lager'):
            source = source_data[i]; srcp = search.predict_proba(source[None])[0, target_position]
            candidates = np.concatenate([mix_concentrations(source, source_data[a], cfg['mixture_weights']) for a in anchors])
            candidate_recipes = [(int(beers[a]), weight, {int(beers[a]): weight}) for a in anchors for weight in cfg['mixture_weights']]
            # An exploratory LP extension checks combinations of multiple target beers.
            # Source/anchor missing defect values cannot support a chemical certificate.
            known_anchors = anchors[np.isfinite(C[anchors][:, positions]).all(1)]
            lp_fraction = None
            if np.isfinite(C[i, positions]).all() and len(known_anchors):
                difference = (C[known_anchors][:, positions] - C[i, positions]).T
                lp = linprog(-np.ones(len(known_anchors)), A_ub=np.vstack([difference, np.ones(len(known_anchors))]),
                             b_ub=np.r_[np.zeros(len(positions)), .75], bounds=(0, None), method='highs')
                if lp.success:
                    lp_fraction = float(lp.x.sum())
                    if lp_fraction > 1e-8:
                        for shrink in np.linspace(.05, 1, 20):
                            weights = shrink * lp.x
                            value = source * (1 - weights.sum()) + weights @ source_data[known_anchors]
                            candidates = np.vstack([candidates, value])
                            recipe = {int(beers[a]): float(w) for a, w in zip(known_anchors, weights) if w > 1e-10}
                            candidate_recipes.append((None, float(weights.sum()), recipe))
            lp_rows.append({'repeat': repeat, 'beer': int(beers[i]), 'source_role': role[i],
                            'max_target_mixture_fraction': lp_fraction, 'known_target_anchors': len(known_anchors),
                            'scope': 'reported_concentration_four_linear_constraints_only_no_LOD_or_reaction_certification'})
            known = np.isfinite(C[i, positions]).all()
            chem_ok = exact_nonincrease(candidates, C[i], positions)
            probabilities = search.predict_proba(candidates)[:, target_position]
            gs = sensory.predict(source[None])[0, stale]; rs = consensus.predict(source[None])[0, stale]
            gpred = sensory.predict(candidates)[:, stale]; rpred = consensus.predict(candidates)[:, stale]
            valid = chem_ok & (probabilities >= max(.5, srcp + .01)) & (gpred <= gs + 1e-9) & (rpred <= rs + 1e-9)
            change = np.linalg.norm(generator_prep.transform(candidates) - generator_prep.transform(source[None]), axis=1) / np.sqrt(C.shape[1])
            valid &= change > 1e-8
            good = np.flatnonzero(valid); k = int(good[np.argmin(change[good])]) if len(good) else None
            value = candidates[k].copy() if k is not None else source.copy()
            locked.append({'row': i, 'candidate': value, 'source': source, 'candidate_index': k,
                           'anchor_beer': candidate_recipes[k][0] if k is not None else None,
                           'weight': candidate_recipes[k][1] if k is not None else 0,
                           'recipe': candidate_recipes[k][2] if k is not None else {},
                           'source_prob': srcp, 'candidate_prob': probabilities[k] if k is not None else srcp,
                           'candidates': len(candidates), 'chemical_feasible': int(chem_ok.sum()), 'valid': len(good),
                           'known_defects': bool(known), 'positive_defects': bool(known and (C[i, positions] > 0).all()),
                           'conditional_missing_any_source_input': bool(np.isnan(C[i]).any()),
                           'source_role': role[i], 'status': 'concentration_mixture_proxy_feasible' if k is not None else 'no_feasible_mixture_source_retained'})
        # Lock exact numeric candidates BEFORE fitting independent J classifiers/sensory.
        digest = hashlib.sha256(np.asarray([v['candidate'] for v in locked], dtype='<f8').tobytes()).hexdigest()
        write_json(dest / f'concentration_candidate_lock_{repeat}.json', {'repeat': repeat, 'sha256_float64_candidates': digest,
                                                                       'features': features, 'selection_completed_before_J_fitting': True})
        evaluator = Pipeline([('prep', preprocessing()), ('model', ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3, max_features=.7, random_state=cfg['seed'], n_jobs=2))]).fit(C[J], labels[J])
        judge = tree_model(cfg).fit(C[J], Y[J]); target_j = list(evaluator.classes_).index('Tripel')
        diagnostics.append({'repeat': repeat, 'concentration_search_balanced_accuracy_on_T': balanced_accuracy_score(labels[T], search.predict(C[T])),
                            'concentration_J_balanced_accuracy_on_T': balanced_accuracy_score(labels[T], evaluator.predict(C[T])),
                            'raw_stale_J_mae_on_T': float(np.abs(judge.predict(C[T])[:, stale] - Y[T, stale]).mean())})
        for v in locked:
            i = v['row']; source, value = v['source'], v['candidate']; jp = evaluator.predict_proba(np.stack([source, value]))[:, target_j]
            js = judge.predict(np.stack([source, value]))[:, stale]; predicted = evaluator.predict(value[None])[0]
            feasible = v['candidate_index'] is not None
            accepted = feasible and predicted == 'Tripel' and jp[1] > jp[0] + .01 and js[1] <= js[0] + 1e-9
            mixture_rows.append({'repeat': repeat, 'beer': int(beers[i]), 'source_role': v['source_role'], 'status': v['status'],
                                 'anchor_beer': v['anchor_beer'], 'anchor_volume_fraction': v['weight'], 'n_candidates': v['candidates'],
                                 'target_component_fractions': str(v['recipe']),
                                 'n_chemical_feasible': v['chemical_feasible'], 'n_search_feasible': v['valid'],
                                 'four_source_defects_observed': v['known_defects'], 'four_source_defects_positive': v['positive_defects'],
                                 'conditional_on_other_source_input_imputation': v['conditional_missing_any_source_input'],
                                 'G_source_Tripel_probability': v['source_prob'], 'G_candidate_Tripel_probability': v['candidate_prob'],
                                 'J_source_Tripel_probability': jp[0], 'J_candidate_Tripel_probability': jp[1],
                                 'J_stale_source_raw_0_6': js[0], 'J_stale_candidate_raw_0_6': js[1],
                                 'J_predicts_Tripel': predicted == 'Tripel', 'independent_proxy_support': bool(accepted),
                                 'real_tasting_validated': False, 'LOD_certification': False})
            mixture_profiles.append({'repeat': repeat, 'beer': int(beers[i]), 'status': v['status'], **dict(zip(features, value))})
    save_csv(pd.DataFrame(frozen_rows), dest / 'frozen_stage3_candidates_raw_score_check.csv')
    save_csv(pd.DataFrame(mixture_rows), dest / 'concentration_mixture_evaluation.csv')
    save_csv(pd.DataFrame(mixture_profiles), dest / 'concentration_mixture_profiles.csv')
    save_csv(pd.DataFrame(diagnostics), dest / 'concentration_holdout_diagnostics.csv')
    save_csv(pd.DataFrame(lp_rows), dest / 'mixture_lp_feasibility.csv')
    write_json(dest / 'status.json', {'complete': True, 'concentration_features': features, 'frozen_candidates_evaluated': len(frozen_rows),
                                     'mixture_records': len(mixture_rows), 'physical_experiments_performed': False})
