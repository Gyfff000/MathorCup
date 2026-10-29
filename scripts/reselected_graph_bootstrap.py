"""Exploratory grouped-bootstrap full graph hyperparameter reselection."""
from pathlib import Path
import os,sys,argparse,json
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[n]='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.linalg import solve
from sklearn.model_selection import GroupKFold
from src.data import find_data,load_data,NON_MOLECULES
from src.common import preprocessing,save_csv,write_json,sha256
from src.stage3_graph import train_correlation,solve_coeff,candidates
from src.q2 import signed_laplacian,top_indices

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir');p.add_argument('--out',default='outputs/reselected_graph_20261029');p.add_argument('--replicates',type=int,default=30);a=p.parse_args(); out=ROOT/a.out
 if out.exists() and any(out.iterdir()): raise SystemExit('Choose a new empty output directory')
 out.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'config_stage3.json').read_text(encoding='utf8'));data=load_data(find_data(a.data_dir));X=data.X.to_numpy();beers=data.meta.beer.to_numpy();targets=list(data.Y);fields=list(data.X);core=np.array([j for j,t in enumerate(targets) if t.startswith(('A_','F_'))]);eligible=np.array([j for j,f in enumerate(fields) if f not in NON_MOLECULES]);Y=pd.read_csv(ROOT/'outputs/stage4/audit/observed_panel_means_0_6.csv').set_index('beer').loc[beers,targets].to_numpy(); original=pd.read_csv(ROOT/'outputs/stage4/raw_graph/q2/top3_with_evidence.csv');choices=candidates(cfg,'real');rng=np.random.default_rng(cfg['seed']+2000);freq={};rows=[];ret=[]
 for b in range(a.replicates):
  ii=rng.integers(len(X),size=len(X)); score=np.zeros(len(choices))
  for train,val in GroupKFold(3).split(X[ii],Y[ii],beers[ii]):
   prep=preprocessing().fit(X[ii[train]]); z=prep.transform(X[ii[train]]);v=prep.transform(X[ii[val]]);mean=Y[ii[train]].mean(0);lhs=z.T@z;rhs=z.T@(Y[ii[train]]-mean);eye=np.eye(z.shape[1]);corr=train_correlation(z);layers={t:signed_laplacian(corr,t) for t in cfg['graph_thresholds']}
   for k,(alpha,lam,tau) in enumerate(choices):
    beta=solve(lhs+alpha*eye+(lam*layers[tau] if lam else 0),rhs,assume_a='pos',check_finite=False);pred=v@beta+mean;score[k]+=np.abs(Y[ii[val]][:,core]-pred[:,core]).sum()/(len(ii)*len(core))
  best=min(range(len(choices)),key=lambda k:(score[k],choices[k][1],-choices[k][0]));alpha,lam,tau=choices[best];z=preprocessing().fit_transform(X[ii]);corr=train_correlation(z);fixed=solve_coeff(z,Y[ii],signed_laplacian(corr,.7),1000,0);tuned=solve_coeff(z,Y[ii],signed_laplacian(corr,tau),alpha,lam)
  for j,t in enumerate(targets):
   aa=top_indices(tuned[:,j],eligible);bb=top_indices(fixed[:,j],eligible);ret.append({'replicate':b,'target':t,'retention':len(set(aa)&set(bb))/3})
   for name,indices in [('reselected',aa),('fixed',bb)]:
    for pos in indices:freq[(name,t,fields[pos])]=freq.get((name,t,fields[pos]),0)+1
  rows.append({'replicate':b,'alpha':alpha,'lambda':lam,'tau':tau,'inner_core_mae':float(score[best])});print('full graph reselection',b+1,flush=True)
 result=[{'target':r.target,'rank':r.rank,'field':r.field,'fixed_frequency':freq.get(('fixed',r.target,r.field),0)/a.replicates,'reselected_frequency':freq.get(('reselected',r.target,r.field),0)/a.replicates} for r in original.itertuples()]
 save_csv(pd.DataFrame(rows),out/'selected_parameters.csv');save_csv(pd.DataFrame(ret),out/'paired_retention.csv');save_csv(pd.DataFrame(result),out/'frequency_comparison.csv');write_json(out/'manifest.json',{'repetitions':a.replicates,'bootstrap':'sample beer with replacement, group repeated copies by beer in inner 3-fold','grid':'stage3 candidates alpha lambda tau','selection':'train-fold preprocessing and Spearman graph, core30 MAE','fixed_params':{'alpha':1000,'lambda':0,'tau':.7},'code_sha256':sha256(Path(__file__))});print(pd.DataFrame(rows).groupby(['alpha','lambda','tau']).size().to_string());print('mean retention',pd.DataFrame(ret).retention.mean())
if __name__=='__main__':main()
