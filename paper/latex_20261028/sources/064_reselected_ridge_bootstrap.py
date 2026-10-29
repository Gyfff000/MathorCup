"""Pilot of bootstrap stability with grouped inner alpha reselection."""
from pathlib import Path
import os,sys,argparse,json
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[n]='2'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from src.data import find_data,load_data,NON_MOLECULES
from src.common import preprocessing,save_csv,write_json,sha256
from src.stage4_models import raw_top3

def fit(X,Y,alpha): return Pipeline([('prep',preprocessing()),('model',Ridge(alpha=alpha))]).fit(X,Y)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir');p.add_argument('--out',default='outputs/reselected_bootstrap_20261029');p.add_argument('--replicates',type=int,default=50);a=p.parse_args(); out=ROOT/a.out
 if out.exists() and any(out.iterdir()): raise SystemExit('Choose a new empty output directory')
 out.mkdir(parents=True,exist_ok=True); cfg=json.loads((ROOT/'config_stage4.json').read_text(encoding='utf8')); data=load_data(find_data(a.data_dir)); X=data.X.to_numpy(); beers=data.meta.beer.to_numpy(); targets=list(data.Y); fields=list(data.X); core=np.array([j for j,t in enumerate(targets) if t.startswith(('A_','F_'))]); Y=pandas_mean=dataY=pd.read_csv(ROOT/'outputs/stage4/audit/observed_panel_means_0_6.csv').set_index('beer').loc[beers,targets].to_numpy(); original=pd.read_csv(ROOT/'outputs/stage4/raw_graph/q2/top3_with_evidence.csv')
 rng=np.random.default_rng(cfg['seed']+999); rows=[]; freq={}; same=[]
 for b in range(a.replicates):
  ii=rng.integers(len(X),size=len(X)); group=beers[ii]; errs={alpha:[] for alpha in cfg['ridge_alphas']}
  for train,val in GroupKFold(n_splits=3).split(X[ii],Y[ii],group):
   prep=preprocessing().fit(X[ii[train]]); xt=prep.transform(X[ii[train]]); xv=prep.transform(X[ii[val]])
   for alpha in errs:
    pred=Ridge(alpha=alpha).fit(xt,Y[ii[train]]).predict(xv); errs[alpha].append(np.abs(pred[:,core]-Y[ii[val]][:,core]).mean())
  alpha=min(errs,key=lambda z:np.mean(errs[z])); tuned=fit(X[ii],Y[ii],alpha); fixed=fit(X[ii],Y[ii],1000)
  for mode,m in [('reselected',tuned),('fixed_1000',fixed)]:
   for target,j,_,rank in raw_top3(m,fields,targets): freq[(mode,target,fields[j])]=freq.get((mode,target,fields[j]),0)+1
  a1=raw_top3(tuned,fields,targets); a2=raw_top3(fixed,fields,targets)
  for j,t in enumerate(targets):
   s1=set(fields[z[1]] for z in a1[3*j:3*j+3]); s2=set(fields[z[1]] for z in a2[3*j:3*j+3]); same.append({'replicate':b,'target':t,'retention':len(s1&s2)/3})
  rows.append({'replicate':b,'selected_alpha':alpha,**{f'mae_alpha_{z}':float(np.mean(v)) for z,v in errs.items()}}); print('reselected bootstrap',b+1,flush=True)
 outrows=[]
 for r in original.itertuples():
  outrows.append({'target':r.target,'rank':r.rank,'field':r.field,'fixed_frequency_50':freq.get(('fixed_1000',r.target,r.field),0)/a.replicates,'reselected_frequency_50':freq.get(('reselected',r.target,r.field),0)/a.replicates})
 save_csv(pd.DataFrame(rows),out/'parameters.csv');save_csv(pd.DataFrame(outrows),out/'top3_frequency_comparison.csv');save_csv(pd.DataFrame(same),out/'paired_retention.csv');write_json(out/'manifest.json',{'replicates':a.replicates,'selection':'bootstrap beers, 3-fold GroupKFold by original beer, training-only preprocessing; select alpha on core30 MAE','alpha_candidates':cfg['ridge_alphas'],'fixed_alpha':1000,'graph_lambda':0,'scope':'ridge alpha sensitivity only; graph lambda and threshold are not reselected','code_sha256':sha256(Path(__file__))}); print(pd.DataFrame(rows).selected_alpha.value_counts().to_string());print('mean paired retention',pd.DataFrame(same).retention.mean());print('mean absolute frequency change',np.mean([abs(z['fixed_frequency_50']-z['reselected_frequency_50']) for z in outrows]))
if __name__=='__main__': main()
