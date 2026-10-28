"""Analyze real future blinded paired ratings; refuse empty/simulated templates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
from src.stage4_trial import analyze_paired_trial
from src.common import save_csv
parser = argparse.ArgumentParser()
parser.add_argument('--ratings', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
result = analyze_paired_trial(pd.read_csv(args.ratings))
save_csv(result, args.out)
print('Analysis uses independent participant means; repeated batches do not increase the participant count.')
