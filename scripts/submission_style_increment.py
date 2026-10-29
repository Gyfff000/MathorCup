"""Exploratory matched ET comparisons: known style alone versus style plus chemistry."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import load_data,find_data
from src.common import save_csv,write_json,sha256
from src.stage3_prediction import tree_model
from src.review_diagnostics import conditional_paired_interval

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir');p.add_argument('--out',default='outputs/submission_style_increment_20261029');a=p.parse_args();out=ROOT/a.out
 if out.exists() and any(out.iterdir()):raise SystemExit('choose an empty output directory')
 out.mkdir(parents=True,exist_ok=True);D=load_data(find_data(a.data_dir));cfg=json.loads((ROOT/'config_stage4.json').read_text(encoding='utf8'));beers=D.meta.beer.to_numpy();styles=D.meta.tasting_category_fine.to_numpy();targets=list(D.Y);X=D.X.to_numpy();Y=pd.read_csv(ROOT/'outputs/stage4/audit/observed_panel_means_0_6.csv').set_index('beer').loc[beers,targets].to_numpy();onehot=pd.get_dummies(styles,dtype=float).to_numpy();inputs={'style_ET':onehot,'style_chemistry_ET':np.column_stack([onehot,X])};oof={k:np.zeros_like(Y) for k in inputs};count=np.zeros(len(X));roles=pd.read_csv(ROOT/'outputs/stage4/prediction/outer_roles.csv')
 for fold,(train,test) in enumerate(RepeatedStratifiedKFold(n_splits=cfg['outer_folds'],n_repeats=cfg['outer_repeats'],random_state=cfg['seed']).split(X,styles)):
  assert set(beers[test])==set(roles.loc[roles.fold==fold,'beer'])
  for name,v in inputs.items():oof[name][test]+=tree_model(cfg).fit(v[train],Y[train]).predict(v[test])
  count[test]+=1;print('matched style/chemistry fold',fold+1,flush=True)
 assert np.all(count==cfg['outer_repeats']);oof={k:v/count[:,None] for k,v in oof.items()};oof['style_median']=pd.read_csv(ROOT/'outputs/review_20261028/oof_style_median.csv').set_index('beer').loc[beers,targets].to_numpy();rows=[];paired=[];bytarget=[]
 for name,pred in oof.items():
  for scope,cols in [('core30',[i for i,t in enumerate(targets) if t.startswith(('A_','F_'))]),('all50',list(range(len(targets))))]:rows.append({'model':name,'scope':scope,'mae':float(abs(pred[:,cols]-Y[:,cols]).mean())})
  for j,t in enumerate(targets):bytarget.append({'model':name,'target':t,'mae':float(abs(pred[:,j]-Y[:,j]).mean())})
 for scope,cols in [('core30',[i for i,t in enumerate(targets) if t.startswith(('A_','F_'))]),('all50',list(range(len(targets))))]:
  for comparator in ['style_ET','style_median']:
   diff=(abs(oof['style_chemistry_ET'][:,cols]-Y[:,cols])-abs(oof[comparator][:,cols]-Y[:,cols])).mean(1);paired.append({'scope':scope,'comparison':'style_chemistry_ET_minus_'+comparator,**conditional_paired_interval(diff,styles)})
 save_csv(pd.DataFrame(rows),out/'metrics.csv');save_csv(pd.DataFrame(paired),out/'paired.csv');save_csv(pd.DataFrame(bytarget),out/'by_target.csv')
 for name,arr in oof.items():save_csv(pd.concat([D.meta[['beer']],pd.DataFrame(arr,columns=targets)],axis=1),out/f'oof_{name}.csv')
 write_json(out/'manifest.json',{'design':'exploratory after review; same ET hyperparameters, outer folds and known one-hot style; compare style only and style plus chemical, each train-fold preprocessing; style-median retained; fixed OOF beer bootstrap, no retraining uncertainty','data_sha256':sha256(ROOT/'outputs/stage4/audit/observed_panel_means_0_6.csv'),'code_sha256':sha256(Path(__file__)),'note':'max_features=.8 operates on different dimensionalities; not a fully matched optimization budget'});print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(paired).to_string(index=False))
if __name__=='__main__':main()
