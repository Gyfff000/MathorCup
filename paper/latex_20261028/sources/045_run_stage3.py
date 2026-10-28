"""Run saved, explicitly scoped third-stage experiments; AI-assisted, 2026-10-25."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import json
from pathlib import Path
from src.common import environment, write_json, sha256
from src.data import load_data, find_data

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('part',choices=['q1','q2','q3'])
    parser.add_argument('--data-dir')
    parser.add_argument('--out',default='outputs/stage3')
    args=parser.parse_args()
    cfg=json.loads(Path('config_stage3.json').read_text(encoding='utf-8'))
    data=load_data(find_data(args.data_dir)); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    expected={r['file']:r['sha256'] for r in __import__('pandas').read_csv('outputs/stage2/audit/source_manifest.csv').to_dict('records')}
    for file in ['1.csv','2.csv','3.xlsx','4.csv','8.xlsx']:
        if sha256(data.root/file)!=expected[file]: raise ValueError('Original source version changed: '+file)
    if args.part=='q1':
        from src.stage3_prediction import run_prediction
        run_prediction(data,out,cfg)
    elif args.part=='q2':
        from src.stage3_graph import run_graph
        run_graph(data,out,cfg)
    else:
        from src.stage3_generation import run_generation
        run_generation(data,out,cfg)
    write_json(out/f'{args.part}_run_manifest.json',{**environment(),'config':cfg,'config_sha256':sha256('config_stage3.json'),'sources_verified':expected})
