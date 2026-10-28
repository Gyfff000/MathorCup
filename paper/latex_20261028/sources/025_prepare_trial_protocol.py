"""Create a future-trial data header and explicitly assumed power sensitivity."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.common import save_csv
from src.stage4_trial import planned_noninferiority_power
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'outputs/stage4/protocol'; out.mkdir(parents=True, exist_ok=True)
targets = pd.read_csv(ROOT / 'outputs/stage4/audit/observed_panel_means_0_6.csv', nrows=0).columns[1:].tolist()
save_csv(pd.DataFrame(columns=['panelist', 'batch', 'condition', 'blind_code', 'session_date', 'style_Tripel_fit'] + targets), out / 'blank_panel_rating_template.csv')
rows = []
for sigma in [.4, .6, .8, 1., 1.2]:
    for margin in [.25, .5]:
        n = next((n for n in range(3, 501) if planned_noninferiority_power(n, sigma, margin) >= .8), None)
        rows.append({'assumed_SD_of_person_mean_paired_difference': sigma, 'noninferiority_margin': margin,
                     'one_sided_alpha_per_defect': .01, 'target_power': .8, 'minimum_independent_panelists': n,
                     'power_at_30': planned_noninferiority_power(30, sigma, margin), 'status': 'planning_assumptions_not_observed_trial_data'})
save_csv(pd.DataFrame(rows), out / 'sample_size_sensitivity.csv')
print('Blank template and assumed power sensitivity saved; no actual tasting data generated.')
