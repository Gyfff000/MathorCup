"""Train-fold-only synthetic missingness stress, not observed laboratory LOD."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import load_data,find_data
from src.stage3_prediction import tree_model
from src.common import save_csv,write_json,sha256

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir');ap.add_argument('--out',default='outputs/missingness_mechanisms_20261029');a=ap.parse_args();out=ROOT/a.out
 if out.exists() and any(out.iterdir()):raise SystemExit('Choose a new empty output directory')
 out.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'config_stage4.json').read_text(encoding='utf8'));d=load_data(find_data(a.data_dir));X=d.X.to_numpy();beers=d.meta.beer.to_numpy();styles=d.meta.tasting_category_fine.to_numpy();targets=list(d.Y);Y=pd.read_csv(ROOT/'outputs/stage4/audit/observed_panel_means_0_6.csv').set_index('beer').loc[beers,targets].to_numpy();roles=pd.read_csv(ROOT/'outputs/stage4/prediction/outer_roles.csv');names=['observed','random_cell_10pct','low_value_below_training_q10','fixed_field_block_10pct'];summed={name:np.zeros_like(Y) for name in names};counts=np.zeros(len(X));audit=[]
 for fold,(train,test) in enumerate(RepeatedStratifiedKFold(n_splits=3,n_repeats=5,random_state=cfg['seed']).split(X,styles)):
  assert set(beers[test])==set(roles.loc[roles.fold==fold,'beer']);model=tree_model(cfg).fit(X[train],Y[train]);rng=np.random.default_rng(cfg['seed']+fold);base=X[test].copy();present=np.isfinite(base);random=(rng.random(base.shape)<.1)&present
  with np.errstate(all='ignore'):q=np.nanquantile(X[train],.1,axis=0)
  low=present & (base<=q[None,:]);block=np.zeros_like(present); selected=rng.choice(X.shape[1],round(X.shape[1]*.1),replace=False);block[:,selected]=present[:,selected]
  for name,mask in [('observed',np.zeros_like(present)),('random_cell_10pct',random),('low_value_below_training_q10',low),('fixed_field_block_10pct',block)]:
   v=base.copy();v[mask]=np.nan;summed[name][test]+=model.predict(v);audit.append({'fold':fold,'mechanism':name,'new_missing_fraction_of_observed':float(mask.sum()/present.sum()),'new_missing_cells':int(mask.sum()),'field_block':';'.join(d.X.columns[selected]) if name=='fixed_field_block_10pct' else ''})
  counts[test]+=1;print('missingness fold',fold+1,flush=True)
 assert np.all(counts==5);results=[];bytarget=[]
 for name,z in summed.items():
  pred=z/counts[:,None];ref=summed['observed']/counts[:,None];err=np.abs(pred-Y)
  for scope,cols in [('core30',[j for j,t in enumerate(targets) if t.startswith(('A_','F_'))]),('all50',list(range(50)))]:results.append({'mechanism':name,'scope':scope,'mae':float(err[:,cols].mean()),'mae_change_vs_observed':float((err[:,cols]-np.abs(ref-Y)[:,cols]).mean()),'mean_abs_prediction_change':float(np.abs(pred[:,cols]-ref[:,cols]).mean())})
  for j,t in enumerate(targets):bytarget.append({'mechanism':name,'target':t,'mae':float(err[:,j].mean()),'change_vs_observed':float((err[:,j]-np.abs(ref[:,j]-Y[:,j])).mean())})
 save_csv(pd.DataFrame(results),out/'metrics.csv');save_csv(pd.DataFrame(bytarget),out/'by_target.csv');save_csv(pd.DataFrame(audit),out/'fold_missingness.csv');write_json(out/'manifest.json',{'design':'synthetic test-only stress; q10 from outer training values; 10% random fields selected per fold and dropped in all held-out beers; fixed 200-tree ET and shared outer splits; no retraining for stress','physical_LOD_known':False,'code_sha256':sha256(Path(__file__))});print(pd.DataFrame(results).to_string(index=False));print(pd.DataFrame(audit).groupby('mechanism').new_missing_fraction_of_observed.mean().to_string())
if __name__=='__main__':main()
