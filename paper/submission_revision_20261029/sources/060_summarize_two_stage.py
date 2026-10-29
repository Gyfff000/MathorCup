"""Paired beer-level bootstrap of nonzero-grade recognition, conditional on saved OOF predictions."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import save_csv

parser = argparse.ArgumentParser()
parser.add_argument('--chemical', required=True)
parser.add_argument('--result-dir', default=str(ROOT / 'outputs/stage5_two_stage_20261028'))
args = parser.parse_args()
out = Path(args.result_dir)
base = pd.read_csv(ROOT / 'outputs/stage4/ordinal/oof_rating_distributions.csv')
candidate = pd.read_csv(out / 'oof_distribution.csv')
styles = pd.read_csv(args.chemical, usecols=['beer', 'tasting_category_fine'])
joined = base[['beer', 'target', 'observed_panel_modal_grade', 'predicted_modal_grade']].merge(
    candidate[['beer', 'target', 'predicted_modal_grade']], on=['beer', 'target'], suffixes=('_base', '_new'), validate='one_to_one')
joined = joined.merge(styles, on='beer', validate='many_to_one')
rows = []
matched_path = out / 'oof_frequency_matched.csv'
if matched_path.exists():
    matched = pd.read_csv(matched_path, usecols=['beer', 'target', 'predicted_modal_grade']).rename(columns={'predicted_modal_grade': 'predicted_modal_grade_matched'})
    joined = joined.merge(matched, on=['beer', 'target'], validate='one_to_one')
comparators = ['base'] + (['matched'] if matched_path.exists() else [])
for scope, frame in [('core30', joined[joined.target.str.startswith(('A_', 'F_'))]), ('all50', joined)]:
  for comparator in comparators:
    frame = frame[frame.observed_panel_modal_grade > 0].copy()
    frame['difference'] = (frame.predicted_modal_grade_new == frame.observed_panel_modal_grade).astype(int) - (frame[f'predicted_modal_grade_{comparator}'] == frame.observed_panel_modal_grade).astype(int)
    per = frame.groupby(['beer', 'tasting_category_fine']).agg(difference=('difference', 'sum'), denominator=('difference', 'size')).reset_index()
    rng = np.random.default_rng(20261028); numerator = np.zeros(5000); denominator = np.zeros(5000)
    # Include beers with no nonzero records in their style as zero-contribution sampling units.
    per = styles.merge(per[['beer', 'difference', 'denominator']], on='beer', how='left').fillna({'difference': 0, 'denominator': 0})
    for _, g in per.groupby('tasting_category_fine'):
        index = rng.integers(len(g), size=(5000, len(g)))
        numerator += g.difference.to_numpy()[index].sum(axis=1)
        denominator += g.denominator.to_numpy()[index].sum(axis=1)
    draws = numerator[denominator > 0] / denominator[denominator > 0]
    low, high = np.quantile(draws, [.025, .975])
    rows.append({'scope': scope, 'comparator': comparator, 'nonzero_beer_target_records': len(frame), 'difference_two_stage_minus_frequency': float(frame.difference.mean()),
                 'conditional_low95': float(low), 'conditional_high95': float(high), 'resamples': 5000,
                 'unit': 'beer stratified by style; fixed OOF predictions'})
save_csv(pd.DataFrame(rows), out / 'paired_nonzero_accuracy.csv')
print(pd.DataFrame(rows).to_string(index=False))
