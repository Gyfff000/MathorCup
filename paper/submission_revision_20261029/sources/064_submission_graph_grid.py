"""Recover full-data graph tuning scores, using the original inner split and grid."""
from pathlib import Path
import os,sys,json,argparse
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.linalg import solve
from sklearn.model_selection import KFold
from src.data import load_data,find_data
from src.stage3_graph import candidates,train_correlation
from src.q2 import signed_laplacian
from src.common import preprocessing,save_csv,write_json,sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir');a=p.parse_args();d=load_data(find_data(a.data_dir));cfg=json.loads((ROOT/'config_stage3.json').read_text(encoding='utf8'));cfg.update(json.loads((ROOT/'config_stage4.json').read_text(encoding='utf8')));X=d.X.to_numpy();Y=pd.read_csv(ROOT/'outputs/stage4/audit/observed_panel_means_0_6.csv').set_index('beer').loc[d.meta.beer,list(d.Y)].to_numpy();core=[i for i,t in enumerate(d.Y) if t.startswith(('A_','F_'))];choices=candidates(cfg,'real');scores=np.zeros(len(choices));seed=cfg['seed']+500
 for train,test in KFold(3,shuffle=True,random_state=seed).split(X):
  prep=preprocessing().fit(X[train]);z=prep.transform(X[train]);v=prep.transform(X[test]);corr=train_correlation(z);layers={t:signed_laplacian(corr,t) for t in cfg['graph_thresholds']};lhs=z.T@z;rhs=z.T@(Y[train]-Y[train].mean(0));eye=np.eye(z.shape[1])
  for k,(alpha,lam,tau) in enumerate(choices):
   beta=solve(lhs+alpha*eye+(lam*layers[tau] if lam else 0),rhs,assume_a='pos',check_finite=False);pred=v@beta+Y[train].mean(0);scores[k]+=np.abs(Y[test][:,core]-pred[:,core]).sum()/(len(Y)*len(core))
 best=min(range(len(choices)),key=lambda i:(scores[i],choices[i][1],-choices[i][0]));rows=[{'alpha':aa,'lambda':ll,'tau':tt,'inner_mae':scores[i],'selected':i==best} for i,(aa,ll,tt) in enumerate(choices)];out=ROOT/'outputs/submission_review_20261029';save_csv(pd.DataFrame(rows),out/'full_graph_tuning_grid.csv');write_json(out/'full_graph_tuning_manifest.json',{'split':'original full-data 3-fold KFold seed stage4_seed+500; scoring core30 MAE','target':'audited panel means exported CSV','code_sha256':sha256(Path(__file__))});print(pd.DataFrame(rows).sort_values('inner_mae').head(8).to_string(index=False));print('selected',choices[best])
if __name__=='__main__':main()
