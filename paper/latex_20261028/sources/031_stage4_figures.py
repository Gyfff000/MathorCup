"""Scientific figures for the observed-score revision; no synthetic observations."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'outputs/stage4'; DEST = OUT / 'figures'; DEST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 210})

def finish(fig, name):
    fig.tight_layout(); fig.savefig(DEST / name, bbox_inches='tight'); plt.close(fig)

metrics = pd.read_csv(OUT / 'prediction/metrics_by_scope.csv')
names = ['mean', 'ridge', 'extra_trees', 'gbr', 'fixed_half_blend', 'concentration_ET', 'standardized_ET_train_isotonic']
labels = ['Mean', 'Ridge', 'ET', 'GBR', 'Half blend', '61-field ET', 'ET + isotonic']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, scope in zip(axes, ['core30', 'all50']):
    a = metrics[metrics.scope == scope].set_index('model').reindex(names)
    ax.bar(np.arange(len(a)), a.mae, color=['#888888', '#7C9BB8', '#4B8BB5', '#BF7277', '#4F8B70', '#B5976F', '#907AAB'])
    ax.set_xticks(np.arange(len(a)), labels, rotation=32, ha='right'); ax.set_ylabel('MAE on observed 0-6 panel means'); ax.set_title(scope)
finish(fig, 'absolute_prediction.png')

ordinal = pd.read_csv(OUT / 'ordinal/metrics_by_scope.csv')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for k, scope in enumerate(['core30', 'all50']):
    a = ordinal[ordinal.scope == scope].set_index('method')
    for offset, (method, name, color) in enumerate([('training_population_distribution', 'Population baseline', '#999999'), ('ET_rating_distribution', 'ET rating distribution', '#4B8BB5')]):
        distance_column = 'cdf_distance' if 'cdf_distance' in a.columns else 'RPS'  # legacy RPS field was a distribution distance
        axes[0].bar(k + (offset - .5) * .28, a.loc[method, distance_column], width=.27, label=name if k == 0 else None, color=color)
        axes[1].bar(k + (offset - .5) * .28, a.loc[method, 'nonzero_panel_mode_accuracy'], width=.27, label=name if k == 0 else None, color=color)
axes[0].set_ylabel('Squared cumulative distribution distance'); axes[1].set_ylabel('Accuracy among nonzero observed modes')
for ax in axes: ax.set_xticks([0, 1], ['core30', 'all50']); ax.legend(fontsize=8)
finish(fig, 'ordinal_distribution.png')

errors = pd.read_csv(OUT / 'audit/sensory_reconstruction_errors.csv')
fig, ax = plt.subplots(figsize=(10, 2.7))
ax.plot(np.arange(1, 51), np.maximum(errors.max_error_without_fitted_replacement, 1e-12), '.', label='Before constant-cell reconstruction', color='#B56766')
ax.plot(np.arange(1, 51), np.maximum(errors.max_error_after_conditional_reconstruction, 1e-12), '.', label='After conditional reconstruction', color='#4F8B70')
ax.set_yscale('log'); ax.set_xlabel('Sensory attribute index (all 50 retained)'); ax.set_ylabel('Maximum absolute discrepancy'); ax.legend(fontsize=8)
finish(fig, 'source_reconciliation.png')

top = pd.read_csv(OUT / 'raw_graph/q2/top3_with_evidence.csv').rename(columns={
    'bootstrap_frequency_conditional_on_full_selected_params': 'bootstrap_frequency_fixed_alpha',
    'within_style_delta_on_selected_folds': 'within_style_delta_mae'})
top['descriptive_support'] = top.evidence_status.str.startswith('comparatively')
fig, ax = plt.subplots(figsize=(7.8, 3.1))
for flag, name, color in [(False, 'Remaining candidates', '#BBBBBB'), (True, 'Descriptive support', '#4F8B70')]:
    a = top[top.descriptive_support == flag]
    ax.scatter(a.bootstrap_frequency_fixed_alpha, a.within_style_delta_mae, color=color, s=15, alpha=.8, label=name)
ax.axhline(0, color='#777777', linewidth=.7); ax.axvline(.5, color='#777777', linewidth=.7, linestyle='--'); ax.set_xlabel('Bootstrap Top-3 frequency (fixed selected alpha)'); ax.set_ylabel('Held-out within-style permutation delta MAE'); ax.legend(fontsize=8)
finish(fig, 'raw_factor_support.png')

oav = pd.read_csv(OUT / 'audit/reference_oav_sensitivity.csv'); oav = oav[oav.threshold_multiplier == 1]
fig, ax = plt.subplots(figsize=(8.7, 3.1)); fs = ['DMS', 'X2.3.butanedione', 'X4VG', 'X.E..2.nonenal']
arrays = [np.log10(a[a > 0]) for f in fs for a in [oav[oav.field == f].reference_oav.dropna().to_numpy()]]
ax.boxplot(arrays, tick_labels=['DMS', 'Diacetyl', '4VG', 'E-2-nonenal'], showfliers=True, flierprops={'markersize': 2})
ax.axhline(0, color='#B56766', linestyle='--', linewidth=1); ax.set_ylabel('log10(reference OAV), positive reports only')
ax.set_title('Reference thresholds are conditional; assay LOD and matrix are unresolved', fontsize=9)
finish(fig, 'reference_oav.png')

transfer = pd.read_csv(OUT / 'transfer/frozen_stage3_candidates_raw_score_check.csv'); transfer = transfer[transfer.radius == 1]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3))
for ax, data, title in [(axes[0], transfer, 'All source roles: 55 repeated records'), (axes[1], transfer[transfer.source_role == 'T_real_holdout'], 'T sources: 10 records from 7 distinct beers')]:
    values = [len(data), int(data.original_search_feasible.sum()), int(data.original_proxy_acceptance.sum()), int(data.original_and_raw_proxy_support.sum())]
    ax.bar(np.arange(4), values, color=['#999999', '#7C9BB8', '#B5976F', '#4F8B70']); ax.set_xticks(np.arange(4), ['Source', 'G feasible', 'Original J', '+ raw G/J'], rotation=20)
    ax.set_ylim(0, len(data) * 1.18); ax.set_title(title, fontsize=9)
    for i, v in enumerate(values): ax.text(i, v + len(data) * .025, str(v), ha='center')
finish(fig, 'raw_transfer_recheck.png')

paper = ROOT / 'paper/revised_absolute_20261025'; eq = paper / 'equations'; eq.mkdir(parents=True, exist_ok=True)
for p in (ROOT / 'paper/revised_20261025/equations').glob('*.png'): shutil.copyfile(p, eq / p.name)
formulas = {
    6: r'$C_j(x)=[f_j(x)-q_j,\;f_j(x)+q_j]\cap[0,6]$',
    18: r'$\bar{s}_{bj}=\frac{1}{|T_b|}\sum_{t\in T_b}\left(\frac{1}{n_{bt}}\sum_{r=1}^{n_{bt}}s_{btrj}\right)$',
    19: r'$q_{bjk}=\frac{1}{|T_b|}\sum_{t\in T_b}\frac{1}{n_{bt}}\sum_{r=1}^{n_{bt}}\mathbf{1}(s_{btrj}=k),\quad k=0,\ldots,6$',
    20: r'$\hat{g}^{\rm mode}_{bj}=\arg\max_k\hat{q}_{bjk},\qquad\hat{g}^{\rm med}_{bj}=\min\{k:\sum_{h=0}^{k}\hat{q}_{bjh}\geq0.5\}$',
    21: r'$\mathrm{RPS}=\frac{1}{6nq}\sum_{b,j}\sum_{k=0}^{5}\left(\sum_{h=0}^{k}\hat{q}_{bjh}-\sum_{h=0}^{k}q_{bjh}\right)^2$',
    22: r'$\max_{w\geq0}\;\mathbf{1}^{T}w\quad\mathrm{s.t.}\quad(C_A-c_s\mathbf{1}^{T})_Dw\leq0,\;\mathbf{1}^{T}w\leq0.75$',
    23: r'$c_{\rm mix}=(1-\mathbf{1}^{T}w)c_s+C_Aw$',
    24: r'$U_j=\bar d_j+t_{1-\alpha_j,n-1}\frac{s_{d_j}}{\sqrt{n}},\qquad U_j<\Delta$',
}
for number, formula in formulas.items():
    fig = plt.figure(figsize=(11, 1.1)); fig.text(.5, .5, formula, fontsize=16, ha='center', va='center'); fig.savefig(eq / f'eq_{number:02d}.png', dpi=220, bbox_inches='tight', transparent=False); plt.close(fig)
print('Six result figures and eight updated equation assets generated.')
