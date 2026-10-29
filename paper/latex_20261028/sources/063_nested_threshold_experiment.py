"""Exploratory, nested training-only threshold selection for modal grades."""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data import find_data,load_data
from src.common import save_csv,write_json,sha256
from src.stage3_prediction import tree_model
from src.review_diagnostics import conditional_paired_interval


def select_thresholds(prob, observed, grid):
    """Select nonzero detection thresholds using only inner OOF training predictions."""
    p=1-prob[:,:,0]; labels=observed.argmax(-1)>0; thresholds=np.full(p.shape[1],.5)
    for j in range(p.shape[1]):
        yes=labels[:,j]; no=~yes
        if yes.sum()<10 or no.sum()<10: continue
        scores=np.array([.5*((p[yes,j]>=t).mean()+(p[no,j]<t).mean()) for t in grid])
        winners=np.flatnonzero(np.isclose(scores,scores.max())); thresholds[j]=grid[winners[np.argmin(np.abs(grid[winners]-.5))]]
    return thresholds


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir'); ap.add_argument('--out',default='outputs/nested_threshold_20261029'); args=ap.parse_args()
    out=ROOT/args.out
    if out.exists() and any(out.iterdir()): raise SystemExit('Choose a new empty output directory')
    out.mkdir(parents=True,exist_ok=True); cfg=json.loads((ROOT/'config_stage4.json').read_text(encoding='utf8')); cfg['extra_trees']=32
    data=load_data(find_data(args.data_dir)); X=data.X.to_numpy(); styles=data.meta.tasting_category_fine.to_numpy(); beers=data.meta.beer.to_numpy(); targets=list(data.Y)
    path=ROOT/'outputs/stage4/ordinal/oof_rating_distributions.csv'; source=pd.read_csv(path).set_index(['beer','target']).loc[pd.MultiIndex.from_product([beers,targets])]
    q=source[[f'observed_panel_p{k}' for k in range(7)]].to_numpy().reshape(len(X),len(targets),7); q/=q.sum(-1,keepdims=True)
    probsum=np.zeros_like(q); votes=np.zeros_like(q); counts=np.zeros(len(X)); thresholds=[]; roles=pd.read_csv(ROOT/'outputs/stage4/prediction/outer_roles.csv'); grid=np.arange(.05,.951,.025)
    splits=RepeatedStratifiedKFold(n_splits=3,n_repeats=5,random_state=cfg['seed'])
    for fold,(train,test) in enumerate(splits.split(X,styles)):
        assert set(beers[test])==set(roles.loc[roles.fold==fold,'beer'])
        inner=np.zeros((len(train),len(targets),7))
        for k,(it,iv) in enumerate(StratifiedKFold(n_splits=2,shuffle=True,random_state=cfg['seed']+fold).split(X[train],styles[train])):
            model=tree_model(cfg).fit(X[train[it]],q[train[it]].reshape(len(it),-1)); inner[iv]=model.predict(X[train[iv]]).reshape(len(iv),len(targets),7)
        t=select_thresholds(inner,q[train],grid); pred=tree_model(cfg).fit(X[train],q[train].reshape(len(train),-1)).predict(X[test]).reshape(len(test),len(targets),7)
        pred=np.maximum(pred,0); pred/=pred.sum(-1,keepdims=True); probsum[test]+=pred
        grade=np.where(1-pred[:,:,0]>=t[None,:],1+pred[:,:,1:].argmax(-1),0)
        for k in range(7): votes[test,:,k]+=(grade==k)
        counts[test]+=1; thresholds.extend({'fold':fold,'target':target,'threshold':float(t[j]),'outer_training_nonzero_count':int((q[train,j].argmax(-1)>0).sum())} for j,target in enumerate(targets)); print('nested fold',fold+1,flush=True)
    assert np.all(counts==5); baseline=(probsum/counts[:,None,None]).argmax(-1); adjusted=votes.argmax(-1); obs=q.argmax(-1)
    records=[]; bytarget=[]; paired=[]
    for name,g in [('frequency_mode',baseline),('nested_balanced_threshold_vote',adjusted)]:
      for scope,cols in [('core30',[j for j,t in enumerate(targets) if t.startswith(('A_','F_'))]),('all50',list(range(50)))]:
        y=obs[:,cols]; z=g[:,cols]; nz=y>0; zero=~nz
        records.append({'model':name,'scope':scope,'overall_accuracy':float((z==y).mean()),'nonzero_exact_accuracy':float((z[nz]==y[nz]).mean()),'nonzero_detection_recall':float((z[nz]>0).mean()),'zero_specificity':float((z[zero]==0).mean()),'grade_mae':float(np.abs(z-y).mean()),'nonzero_records':int(nz.sum())})
      for j,t in enumerate(targets):
        nz=obs[:,j]>0; bytarget.append({'model':name,'target':t,'nonzero_count':int(nz.sum()),'nonzero_exact_accuracy':float((g[nz,j]==obs[nz,j]).mean()) if nz.any() else np.nan,'nonzero_detected':float((g[nz,j]>0).mean()) if nz.any() else np.nan,'overall_accuracy':float((g[:,j]==obs[:,j]).mean())})
    for scope,cols in [('core30',[j for j,t in enumerate(targets) if t.startswith(('A_','F_'))]),('all50',list(range(50)))]:
        # Fixed OOF predictions; resample beer units and recompute ratio directly.
        rng=np.random.default_rng(cfg['seed']); groups=[np.flatnonzero(styles==s) for s in np.unique(styles)]; vals=[]
        for _ in range(5000):
            idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]); y=obs[idx][:,cols]; m=y>0; a=adjusted[idx][:,cols]; b=baseline[idx][:,cols]; vals.append(float(((a==y)&m).sum()/m.sum()-((b==y)&m).sum()/m.sum()))
        y=obs[:,cols]; m=y>0; delta=float(((adjusted[:,cols]==y)&m).sum()/m.sum()-((baseline[:,cols]==y)&m).sum()/m.sum()); paired.append({'scope':scope,'metric':'nonzero_exact_accuracy_difference','difference':delta,'bootstrap_lower':float(np.quantile(vals,.025)),'bootstrap_upper':float(np.quantile(vals,.975))})
    save_csv(pd.DataFrame(records),out/'metrics.csv'); save_csv(pd.DataFrame(bytarget),out/'by_target.csv'); save_csv(pd.DataFrame(paired),out/'paired.csv'); save_csv(pd.DataFrame(thresholds),out/'fold_thresholds.csv')
    save_csv(pd.DataFrame({'beer':np.repeat(beers,len(targets)),'target':np.tile(targets,len(beers)),'observed_mode':obs.ravel(),'baseline_mode':baseline.ravel(),'threshold_vote_mode':adjusted.ravel()}),out/'oof_grades.csv')
    write_json(out/'manifest.json',{'method':'exploratory 32-tree ET; each outer training set 2-fold inner OOF threshold tuned for balanced zero/nonzero detection; decision votes across five repeats; no recalibration of probabilities','train_positive_and_negative_min':10,'grid':[.05,.95,.025],'source_sha256':sha256(path),'code_sha256':sha256(Path(__file__)),'uncertainty':'fixed OOF beer-stratified bootstrap, no refit'}); print(pd.DataFrame(records).to_string(index=False)); print(pd.DataFrame(paired).to_string(index=False))
if __name__=='__main__': main()
