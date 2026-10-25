"""Repeat the full graph control experiment on the newly observed raw endpoint."""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '2')
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
from dataclasses import replace
import pandas as pd
from src.common import environment, write_json, sha256
from src.data import load_data, find_data
from src.stage3_graph import run_graph
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(); parser.add_argument('--data-dir'); args = parser.parse_args()
cfg = json.loads((ROOT / 'config_stage3.json').read_text(encoding='utf-8'))
cfg.update(json.loads((ROOT / 'config_stage4.json').read_text(encoding='utf-8')))
data = load_data(find_data(args.data_dir))
panel = pd.read_csv(ROOT / 'outputs/stage4/audit/observed_panel_means_0_6.csv').set_index('beer').reindex(data.meta.beer)
raw_data = replace(data, Y=pd.DataFrame(panel.to_numpy(), columns=data.Y.columns))
destination = ROOT / 'outputs/stage4/raw_graph'
run_graph(raw_data, destination, cfg)
write_json(destination / 'run_manifest.json', {**environment(), 'config': cfg, 'target': 'observed_raw_equal_panelist_mean_0_6',
                                             'code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in [Path(__file__), ROOT / 'src/stage3_graph.py']}})
