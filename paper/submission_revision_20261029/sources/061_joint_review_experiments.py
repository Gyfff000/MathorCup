from pathlib import Path
import os, sys, json, hashlib, argparse
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(n,'2')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from src.data import find_data, load_data, CHEMICAL_DEFECTS, AGGREGATES, NON_MOLECULES
from src.common import preprocessing, save_csv, write_json, sha256
from src.stage3_prediction import tree_model


def risk_mix(source, anchors, threshold):
    d=(anchors-source).T
    b=np.maximum(source,threshold)-source
    scale=np.maximum(np.abs(d).max(axis=1),1e-12)
    result=linprog(-np.ones(len(anchors)),A_ub=np.vstack([d/scale[:,None],np.ones(len(anchors))]),b_ub=np.r_[b/scale,.75],bounds=(0,None),method='highs')
    if not result.success: raise RuntimeError(result.message)
    return result.x


def main():
    cfg=json.loads((ROOT/'config_stage4.json').read_text(encoding='utf-8'))
    parser=argparse.ArgumentParser(); parser.add_argument('--data-dir'); args=parser.parse_args()
    data=load_data(find_data(args.data_dir)); beers=data.meta.beer.to_numpy(); styles=data.meta.tasting_category_fine.to_numpy(); targets=list(data.Y)
    out=ROOT/'outputs/joint_20261028'; out.mkdir(parents=True,exist_ok=True)
    Y=pd.read_csv(ROOT/'outputs/stage4/audit/observed_panel_means_0_6.csv').set_index('beer').loc[beers,targets].to_numpy()
    features=[f for f in data.X if f not in AGGREGATES]; X=data.X[features].copy(); exceptions=pd.read_csv(ROOT/'outputs/stage4/audit/chemical_source_exceptions.csv')
    assert not exceptions.duplicated(['beer','field']).any()
    index={int(b):i for i,b in enumerate(beers)}; masked=X.copy(); applied=[]
    for row in exceptions.itertuples():
        if row.field in masked.columns:
            i=index[int(row.beer)]; masked.iloc[i,masked.columns.get_loc(row.field)]=np.nan; applied.append((row.beer,row.field))
    registry=pd.read_csv(ROOT/'outputs/stage4/audit/concentration_feature_registry.csv')
    reconciled=pd.read_csv(ROOT/'outputs/stage4/audit/chemical_scale_reconciliation.csv')
    verified=[f for f in features if f in set(registry.field) and f in set(reconciled.loc[reconciled.status=='all_comparable_pairs_reconciled','field'])]
    variants={'original':X,'masked_exceptions':masked,'reconciled_unit_fields':X[verified]}
    preds={k:np.zeros_like(Y) for k in variants}; counts=np.zeros(len(beers)); top={k:[] for k in variants}; rows=[]
    previous=pd.read_csv(ROOT/'outputs/stage4/prediction/outer_roles.csv')
    split=RepeatedStratifiedKFold(n_splits=cfg['outer_folds'],n_repeats=cfg['outer_repeats'],random_state=cfg['seed'])
    for fold,(train,test) in enumerate(split.split(X,styles)):
        assert set(beers[test])==set(previous.loc[previous.fold==fold,'beer'])
        print(f'Chemical sensitivity fold {fold+1}/15',flush=True)
        for name,frame in variants.items():
            a=frame.to_numpy(); model=tree_model(cfg).fit(a[train],Y[train]); pred=model.predict(a[test]); preds[name][test]+=pred
            prep=preprocessing().fit(a[train]); ridge=Ridge(alpha=1000).fit(prep.transform(a[train]),Y[train]); coef=np.abs(ridge.coef_); cols=list(frame)
            valid=np.array([i for i,f in enumerate(cols) if f not in NON_MOLECULES]); ranks=[]
            for j in range(len(targets)): ranks.append(set(cols[i] for i in valid[np.argsort(coef[j,valid])[-3:]]))
            top[name].append(ranks)
        counts[test]+=1
    assert np.all(counts==cfg['outer_repeats'])
    for k in preds: preds[k]/=counts[:,None]
    core=[j for j,t in enumerate(targets) if t.startswith(('A_','F_'))]
    targetrows=[]
    for name,p in preds.items():
        for scope,idx in [('core30',core),('all50',list(range(len(targets))))]:
            err=np.abs(Y[:,idx]-p[:,idx]).mean(axis=1); ref=np.abs(Y[:,idx]-preds['original'][:,idx]).mean(axis=1)
            rows.append({'variant':name,'scope':scope,'fields':variants[name].shape[1],'mae':err.mean(),'paired_mae_change':(err-ref).mean(),'mean_absolute_prediction_change':np.abs(p[:,idx]-preds['original'][:,idx]).mean()})
        for j,t in enumerate(targets):
            overlaps=[len(top[name][f][j]&top['original'][f][j])/3 for f in range(len(top[name]))]
            save={'variant':name,'target':t,'mae':np.abs(Y[:,j]-p[:,j]).mean(),'original_mae':np.abs(Y[:,j]-preds['original'][:,j]).mean(),'mean_top3_retention_fixed_ridge':np.mean(overlaps)}
            targetrows.append(save)
    save_csv(pd.DataFrame(rows),out/'chemical_sensitivity_summary.csv'); save_csv(pd.DataFrame(targetrows),out/'chemical_sensitivity_by_target.csv')
    # Candidate search uses G alone; candidate array is frozen before J is fitted.
    fields=registry.field.tolist(); C=data.calibrated.set_index('beer').loc[beers,fields].where(lambda x:x>=0).to_numpy(); pos=[fields.index(f) for f in CHEMICAL_DEFECTS]; stale=targets.index('stale_hops')
    roles=pd.read_csv(ROOT/'outputs/stage3/q3/disjoint_roles.csv'); allrows=[]; allprofiles=[]; locks=[]
    for repeat in range(5):
        print(f'Joint risk and proxy evaluation repeat {repeat+1}/5',flush=True)
        role=roles[roles.repeat==repeat].set_index('beer').loc[beers,'role'].to_numpy(); G=np.flatnonzero(role=='G_search_training'); J=np.flatnonzero(role=='J_evaluator_training'); T=np.flatnonzero(role=='T_real_holdout')
        fill=preprocessing().fit(C[G]); values=np.where(np.isfinite(C),C,fill.named_steps['impute'].statistics_)
        search=Pipeline([('prep',preprocessing()),('model',LogisticRegression(C=1,max_iter=3000,random_state=cfg['seed']))]).fit(C[G],styles[G]); sensory=tree_model(cfg).fit(C[G],Y[G]); consensus=Pipeline([('prep',preprocessing()),('model',Ridge(alpha=1000))]).fit(C[G],Y[G]); pt=list(search.classes_).index('Tripel')
        anchors=G[(styles[G]=='Tripel') & np.isfinite(C[G][:,pos]).all(axis=1)]; locked=[]
        for i in np.flatnonzero(styles=='Pils/Lager'):
            s=values[i]; base=search.predict_proba(s[None])[0,pt]; gs=sensory.predict(s[None])[0,stale]; rs=consensus.predict(s[None])[0,stale]
            for multiplier in [.5,1.,2.]:
                threshold=multiplier*np.array([19.,.09,.2,.0001]); known=np.isfinite(C[i,pos]).all(); candidates=[]; recipes=[]; maxfraction=0.
                if known and len(anchors):
                    w=risk_mix(C[i,pos],C[anchors][:,pos],threshold); maxfraction=w.sum()
                    for shrink in np.linspace(.05,1,20):
                        ww=w*shrink; candidates.append(s*(1-ww.sum())+ww@values[anchors]); recipes.append({int(beers[a]):float(z) for a,z in zip(anchors,ww) if z>1e-10})
                    for a in anchors:
                        for z in cfg['mixture_weights']:
                            candidates.append(s*(1-z)+values[a]*z); recipes.append({int(beers[a]):float(z)})
                found=None; valid_count=0
                if candidates:
                    V=np.array(candidates); risk=(V[:,pos] <= np.maximum(C[i,pos],threshold)+1e-12).all(axis=1); p=search.predict_proba(V)[:,pt]; sv=sensory.predict(V)[:,stale]; rv=consensus.predict(V)[:,stale]
                    distance=np.linalg.norm(fill.transform(V)-fill.transform(s[None]),axis=1)/np.sqrt(len(fields)); valid=risk & (p>=max(.5,base+.01)) & (sv<=gs+1e-9) & (rv<=rs+1e-9) & (distance>1e-8); good=np.flatnonzero(valid); valid_count=len(good)
                    if len(good): found=int(good[np.argmin(distance[good])])
                value=V[found] if found is not None else s.copy()
                locked.append({'i':i,'multiplier':multiplier,'source':s,'value':value,'feasible':found is not None,'recipe':recipes[found] if found is not None else {},'maxfraction':maxfraction,'valid_count':valid_count,'base':base,'gprob':p[found] if found is not None else base,'gs':gs,'gscore':sv[found] if found is not None else gs,'rs':rs,'rscore':rv[found] if found is not None else rs})
        digest=hashlib.sha256(np.array([v['value'] for v in locked],dtype='<f8').tobytes()).hexdigest(); locks.append({'repeat':repeat,'sha256_float64_candidates':digest,'selection_completed_before_J_fitting':True,'features':fields})
        judge=Pipeline([('prep',preprocessing()),('model',ExtraTreesClassifier(n_estimators=200,min_samples_leaf=3,max_features=.7,random_state=cfg['seed'],n_jobs=2))]).fit(C[J],styles[J]); sensoryj=tree_model(cfg).fit(C[J],Y[J]); pj=list(judge.classes_).index('Tripel')
        for v in locked:
            i=v['i']; z=np.stack([v['source'],v['value']]); prob=judge.predict_proba(z)[:,pj]; score=sensoryj.predict(z)[:,stale]; predicted=judge.predict(z[1:])[0]; accepted=v['feasible'] and predicted=='Tripel' and prob[1]>prob[0]+.01 and score[1]<=score[0]+1e-9
            row={'repeat':repeat,'beer':int(beers[i]),'source_role':role[i],'threshold_multiplier':v['multiplier'],'max_chemical_fraction':v['maxfraction'],'search_feasible':v['feasible'],'search_candidate_count':v['valid_count'],'target_fractions':json.dumps(v['recipe'],sort_keys=True),'G_source_Tripel_probability':v['base'],'G_candidate_Tripel_probability':v['gprob'],'G_source_stale':v['gs'],'G_candidate_stale':v['gscore'],'G_ridge_source_stale':v['rs'],'G_ridge_candidate_stale':v['rscore'],'J_source_Tripel_probability':prob[0],'J_candidate_Tripel_probability':prob[1],'J_predicts_Tripel':predicted=='Tripel','J_source_stale':score[0],'J_candidate_stale':score[1],'J_proxy_support':bool(accepted),'source_has_imputed_concentration':bool(np.isnan(C[i]).any())}
            for k,f in enumerate(CHEMICAL_DEFECTS): row['source_'+f]=C[i,pos[k]]; row['candidate_'+f]=v['value'][pos[k]]
            allrows.append(row); allprofiles.append({'repeat':repeat,'beer':int(beers[i]),'threshold_multiplier':v['multiplier'],'search_feasible':v['feasible'],**dict(zip(fields,v['value']))})
    save_csv(pd.DataFrame(allrows),out/'joint_candidate_evaluation.csv'); save_csv(pd.DataFrame(allprofiles),out/'joint_candidate_profiles.csv'); write_json(out/'candidate_locks.json',locks)
    summary=pd.DataFrame(allrows).groupby(['threshold_multiplier','source_role'],as_index=False).agg(records=('beer','size'),unique_beers=('beer','nunique'),chemical_possible=('max_chemical_fraction',lambda x:int((x>1e-8).sum())),search_feasible=('search_feasible','sum'),J_proxy_support=('J_proxy_support','sum'))
    save_csv(summary,out/'joint_summary_by_role.csv')
    write_json(out/'manifest.json',{'design':'exploratory after review; G candidate selection frozen before fitting J; source role J is in J training; T is internally held out, not external','exception_cells_in_nonaggregate_features':len(applied),'verified_unit_field_count':len(verified),'verified_fields':verified,'ridge_alpha_fixed':1000,'folds':cfg['outer_folds'],'repeats':cfg['outer_repeats'],'physical_experiment':False,'code_sha256':sha256(Path(__file__))})
    print(pd.DataFrame(rows).to_string(index=False)); print(summary.to_string(index=False))

if __name__=='__main__': main()

