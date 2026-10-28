"""Describe saved out-of-fold grades and split-conformal coverage without refitting."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import save_csv, write_json, sha256

parser = argparse.ArgumentParser()
parser.add_argument('--chemical', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
out = Path(args.out)
if out.exists() and any(out.iterdir()):
    parser.error('Choose an empty output directory')
out.mkdir(parents=True, exist_ok=True)
base = ROOT / 'outputs/stage4'
ordinal_path = base / 'ordinal/oof_rating_distributions.csv'
interval_path = base / 'prediction/absolute_panel_mean_intervals.csv'
population_path = base / 'audit/panel_sample_counts.csv'
meta = pd.read_csv(args.chemical, usecols=['beer', 'tasting_category_fine'])
counts = pd.read_csv(population_path)
if meta.beer.duplicated().any() or counts.beer.duplicated().any():
    raise ValueError('Duplicate sample metadata')
ratings = pd.read_csv(ordinal_path).merge(meta, on='beer', validate='many_to_one')
ratings['core'] = ratings.target.str.startswith(('A_', 'F_'))
ratings['absolute_grade_error'] = (ratings.predicted_modal_grade - ratings.observed_panel_modal_grade).abs()
confusion, target_metrics, style_metrics = [], [], []
for scope, group in [('core30', ratings[ratings.core]), ('all50', ratings)]:
    for observed in range(7):
        for predicted in range(7):
            confusion.append({'scope': scope, 'observed_grade': observed, 'predicted_grade': predicted,
                              'beer_target_count': int(((group.observed_panel_modal_grade == observed) &
                                                        (group.predicted_modal_grade == predicted)).sum())})
    for field, destination in [('target', target_metrics), ('tasting_category_fine', style_metrics)]:
        for value, sample in group.groupby(field, sort=True):
            nonzero = sample[sample.observed_panel_modal_grade > 0]
            destination.append({'scope': scope, field: value, 'unique_beers': int(sample.beer.nunique()),
                                'beer_target_count': len(sample), 'nonzero_observed_count': len(nonzero),
                                'overall_modal_accuracy': float((sample.predicted_modal_grade == sample.observed_panel_modal_grade).mean()),
                                'nonzero_modal_accuracy': float((nonzero.predicted_modal_grade == nonzero.observed_panel_modal_grade).mean()) if len(nonzero) else np.nan,
                                'nonzero_predicted_zero_fraction': float((nonzero.predicted_modal_grade == 0).mean()) if len(nonzero) else np.nan,
                                'modal_absolute_grade_error': float(sample.absolute_grade_error.mean())})
intervals = pd.read_csv(interval_path).merge(meta, on='beer', validate='many_to_one').merge(counts, on='beer', validate='many_to_one')
intervals['covered'] = (intervals.observed_panel_mean >= intervals.lower_0_6) & (intervals.observed_panel_mean <= intervals.upper_0_6)
intervals['width'] = intervals.upper_0_6 - intervals.lower_0_6
intervals['panel_group'] = pd.cut(intervals.distinct_panelists, bins=[0, 4, 7, 100], labels=['3-4', '5-7', '8+'])
coverage = []
for grouping in ['target', 'tasting_category_fine', 'panel_group']:
    for value, sample in intervals.groupby(grouping, observed=True, sort=True):
        coverage.append({'grouping': grouping, 'group': str(value), 'unique_beers': int(sample.beer.nunique()),
                         'test_beer_attribute_records': len(sample), 'coverage': float(sample.covered.mean()),
                         'mean_width': float(sample.width.mean())})
save_csv(pd.DataFrame(confusion), out / 'modal_confusion.csv')
save_csv(pd.DataFrame(target_metrics), out / 'modal_by_target.csv')
save_csv(pd.DataFrame(style_metrics), out / 'modal_by_style.csv')
save_csv(pd.DataFrame(coverage), out / 'coverage_by_group.csv')
core = ratings[ratings.core]; nz = core[core.observed_panel_modal_grade > 0]
summary = {'core_beer_target_records': len(core), 'core_nonzero_records': len(nz),
           'core_nonzero_predicted_zero_fraction': float((nz.predicted_modal_grade == 0).mean()),
           'core_nonzero_modal_accuracy': float((nz.predicted_modal_grade == nz.observed_panel_modal_grade).mean()),
           'coverage_repeat_warning': 'Repeated test records share beers; group rates are descriptive, not independent-binomial guarantees.',
           'input_sha256': {p.name: sha256(p) for p in [ordinal_path, interval_path, population_path]}}
write_json(out / 'summary.json', summary)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matrix = pd.DataFrame(confusion).query("scope == 'core30'").pivot(index='observed_grade', columns='predicted_grade', values='beer_target_count').to_numpy()
fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
im = ax.imshow(matrix, cmap='Blues')
for i in range(7):
    for j in range(7): ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > matrix.max() * .55 else 'black', fontsize=8)
ax.set(xlabel='预测面板众数等级', ylabel='观测面板众数等级', xticks=range(7), yticks=range(7), title='核心30项：按啤酒留出预测的等级混淆')
fig.colorbar(im, ax=ax, label='啤酒-属性记录数'); fig.savefig(out / 'ordinal_confusion.png', dpi=220); plt.close(fig)
print(json.dumps(summary, ensure_ascii=False))
