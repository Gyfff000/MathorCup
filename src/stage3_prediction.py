"""Nested shrinkage blending, ordinal sets and robustness; no novelty claim."""
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from .common import preprocessing, TrainQuantileClipper, progress, save_csv, write_json
from .q1 import metric_rows
from .stage2 import conformal_quantile
from .stage3_algorithms import fixed_grade_boundaries, grade_with_fixed_boundaries, ordinal_interval_image, shrunk_blend_weights, paired_oof_bootstrap


def tree_model(cfg,kind='et',clip=.01):
    prep=preprocessing(); prep.set_params(clip=TrainQuantileClipper(clip,1-clip))
    if kind=='et':
        model=ExtraTreesRegressor(n_estimators=cfg['extra_trees'],min_samples_leaf=3,max_features=.8,random_state=cfg['seed'],n_jobs=2)
    else:
        model=MultiOutputRegressor(GradientBoostingRegressor(n_estimators=cfg['gbr_estimators'],learning_rate=.04,max_depth=2,min_samples_leaf=5,max_features=.7,random_state=cfg['seed']),n_jobs=2)
    return Pipeline([('prep',prep),('model',model)])


def fit_pair(X,Y,cfg):
    with joblib.parallel_backend('threading'):
        et=tree_model(cfg).fit(X,Y)
        gb=tree_model(cfg,'gbr').fit(X,Y)
    return et,gb


def predict_pair(pair,X):
    with joblib.parallel_backend('threading'):
        return pair[0].predict(X),pair[1].predict(X)


def learn_blend_nested(X,Y,cfg,core,fold_id):
    """No outer-test values enter this routine. V labels never enter S fits."""
    inner=list(KFold(3,shuffle=True,random_state=cfg['seed']+fold_id).split(X))
    pred_et,pred_gb=np.zeros_like(Y),np.zeros_like(Y)
    validation_predictions={eta:np.zeros_like(Y) for eta in cfg['blend_penalties']}
    for vfold,(s,v) in enumerate(inner):
        pe,pg=np.zeros_like(Y[s]),np.zeros_like(Y[s])
        for a,b in KFold(3,shuffle=True,random_state=cfg['seed']+vfold+99).split(s):
            fitted=fit_pair(X[s[a]],Y[s[a]],cfg)
            pe[b],pg[b]=predict_pair(fitted,X[s[b]])
        fitted=fit_pair(X[s],Y[s],cfg)
        ve,vg=predict_pair(fitted,X[v]); pred_et[v],pred_gb[v]=ve,vg
        for eta in cfg['blend_penalties']:
            w,_=shrunk_blend_weights(Y[s],pe,pg,eta)
            validation_predictions[eta][v]=ve*(1-w)+vg*w
    scores={eta:float(np.abs(Y[:,core]-pred[:,core]).mean()) for eta,pred in validation_predictions.items()}
    best=min(cfg['blend_penalties'],key=lambda eta:(scores[eta],-eta))
    w,glob=shrunk_blend_weights(Y,pred_et,pred_gb,best)
    unshrunk,_=shrunk_blend_weights(Y,pred_et,pred_gb,0)
    return w,unshrunk,glob,best,scores


def run_prediction(data,out,cfg):
    start=time.perf_counter(); dest=Path(out)/'q1'; dest.mkdir(parents=True,exist_ok=True)
    X,Y=data.X.to_numpy(),data.Y.to_numpy(); targets=list(data.Y)
    core=np.array([j for j,t in enumerate(targets) if t.startswith(('A_','F_'))]); labels=data.meta.tasting_category_fine.to_numpy()
    names=['mean','ridge','extra_trees','gbr','fixed_half_blend','global_oof_blend','unshrunk_target_blend','shrunk_target_blend']
    sums={m:np.zeros_like(Y) for m in names}; count=np.zeros(len(Y)); assignments=[]; fold_rows=[]; weight_rows=[]; stress=[]; clipping=[]
    grade_votes=np.zeros((len(Y),Y.shape[1],7),int)
    for fold,(train,test) in enumerate(RepeatedStratifiedKFold(n_splits=cfg['outer_folds'],n_repeats=cfg['outer_repeats'],random_state=cfg['seed']).split(X,labels)):
        progress(f'S3 Q1 fold {fold+1}/15: nested inner weights; elapsed {(time.perf_counter()-start)/60:.1f} min')
        w,wu,glob,eta,eta_scores=learn_blend_nested(X[train],Y[train],cfg,core,fold)
        pair=fit_pair(X[train],Y[train],cfg); etp,gbp=predict_pair(pair,X[test])
        ridge_search=GridSearchCV(Pipeline([('prep',preprocessing()),('model',Ridge())]),{'model__alpha':cfg['ridge_alphas']},
            scoring=lambda est,xx,yy:-np.abs(yy[:,core]-est.predict(xx)[:,core]).mean(),cv=KFold(3,shuffle=True,random_state=cfg['seed']),n_jobs=1)
        ridge=ridge_search.fit(X[train],Y[train]).best_estimator_
        predictions={'mean':np.broadcast_to(Y[train].mean(0),Y[test].shape),'ridge':ridge.predict(X[test]),'extra_trees':etp,'gbr':gbp,
                     'fixed_half_blend':(etp+gbp)/2,'global_oof_blend':etp*(1-glob)+gbp*glob,'unshrunk_target_blend':etp*(1-wu)+gbp*wu,'shrunk_target_blend':etp*(1-w)+gbp*w}
        for name,pred in predictions.items():
            sums[name][test]+=pred
            for scope,ix in [('core30',core),('all50',np.arange(Y.shape[1]))]:
                fold_rows.append({'fold':fold,'repeat':fold//3,'model':name,'scope':scope,'mae':np.abs(Y[test][:,ix]-pred[:,ix]).mean(),'n_test':len(test)})
        count[test]+=1
        assignments.extend({'fold':fold,'repeat':fold//3,'beer':int(data.meta.iloc[i].beer),'role':'outer_validation'} for i in test)
        weight_rows.extend({'fold':fold,'target':target,'weight_GBR':w[j],'global_weight_GBR':glob,'eta':eta} for j,target in enumerate(targets))
        boundaries=fixed_grade_boundaries(Y[train]); grades=grade_with_fixed_boundaries(etp,boundaries)
        for j in range(Y.shape[1]): np.add.at(grade_votes[:,j],(test,grades[:,j]),1)
        rng=np.random.default_rng(cfg['seed']+fold); low=np.nanquantile(X[train],.2,axis=0)
        for scenario in ['MCAR','low_value_preferential']:
            for rate in [.0,.05,.1,.2]:
                eligible=np.flatnonzero(np.isfinite(X[test]).ravel()); n=int(rate*len(eligible)); a=X[test].copy()
                weights=np.where((a<=low).ravel()[eligible],4.,1.) if scenario!='MCAR' else np.ones(len(eligible))
                if n: a.ravel()[rng.choice(eligible,n,replace=False,p=weights/weights.sum())]=np.nan
                pe,pg=predict_pair(pair,a)
                for name,pred in [('extra_trees',pe),('shrunk_target_blend',pe*(1-w)+pg*w)]:
                    stress.append({'fold':fold,'scenario':scenario,'extra_missing_rate':rate,'model':name,'core30_mae':np.abs(Y[test][:,core]-pred[:,core]).mean(),'all50_mae':np.abs(Y[test]-pred).mean()})
        for lower in [0,.01,.025]:
            alternative=pair[0] if lower==.01 else tree_model(cfg,clip=lower).fit(X[train],Y[train])
            pred=alternative.predict(X[test]); clipping.append({'fold':fold,'lower_clip':lower,'core30_mae':np.abs(Y[test][:,core]-pred[:,core]).mean(),'all50_mae':np.abs(Y[test]-pred).mean()})
        # Save progress atomically per finished outer fold.
        save_csv(pd.DataFrame(fold_rows),dest/'fold_metrics.csv'); save_csv(pd.DataFrame(weight_rows),dest/'blend_weights.csv')
        write_json(dest/'progress.json',{'completed_outer_folds':fold+1,'total_outer_folds':15,'elapsed_seconds':time.perf_counter()-start})
    metric=[]
    for name in names:
        pred=sums[name]/count[:,None]
        save_csv(pd.concat([data.meta[['beer']],pd.DataFrame(pred,columns=targets)],axis=1),dest/f'oof_{name}.csv')
        metric+=metric_rows(Y,pred,targets,name)
    bytarget=pd.DataFrame(metric); save_csv(bytarget,dest/'metrics_by_target.csv')
    scopes=[]
    for scope,fields in [('core30',[targets[j] for j in core]),('all50',targets)]:
        a=bytarget[bytarget.target.isin(fields)].groupby('model')[['mae','rmse','r2']].mean().reset_index(); a.insert(0,'scope',scope); scopes.append(a)
    summary=pd.concat(scopes,ignore_index=True); save_csv(summary,dest/'metrics_by_scope.csv')
    save_csv(pd.DataFrame(assignments),dest/'outer_roles.csv'); save_csv(pd.DataFrame(stress),dest/'missingness_stress.csv'); save_csv(pd.DataFrame(clipping),dest/'clipping_ablation.csv')
    save_csv(pd.concat([data.meta[['beer']],pd.DataFrame(grade_votes.argmax(2),columns=targets)],axis=1),dest/'ASSUMPTION_oof_relative_grades.csv')
    paired=[]
    for name in ['gbr','global_oof_blend','unshrunk_target_blend','shrunk_target_blend']:
        for scope,ix in [('core30',core),('all50',np.arange(Y.shape[1]))]:
            r=paired_oof_bootstrap(Y[:,ix],sums[name][:,ix]/count[:,None],sums['extra_trees'][:,ix]/count[:,None],labels)
            r.update(model_A=name,model_B='extra_trees',target_scope=scope); paired.append(r)
    write_json(dest/'paired_conditional_comparison.json',paired)
    intervals=[]; coverage_rows=[]
    for repeat in range(cfg['conformal_repeats']):
        train,rest=train_test_split(np.arange(len(X)),test_size=.4,random_state=cfg['seed']+71+repeat)
        calibration,test=train_test_split(rest,test_size=.5,random_state=cfg['seed']+72+repeat)
        model=tree_model(cfg).fit(X[train],Y[train]); pp=model.predict(X[test]); q=conformal_quantile(np.abs(Y[calibration]-model.predict(X[calibration])),cfg['conformal_alpha'])
        boundaries=fixed_grade_boundaries(Y[train]); lo,hi=ordinal_interval_image(pp-q,pp+q,boundaries); true=grade_with_fixed_boundaries(Y[test],boundaries); point=grade_with_fixed_boundaries(pp,boundaries)
        for j,t in enumerate(targets):
            coverage_rows.append({'repeat':repeat,'target':t,'continuous_coverage':((Y[test,j]>=pp[:,j]-q[j])&(Y[test,j]<=pp[:,j]+q[j])).mean(),'interval_width':2*q[j],
                'relative_grade_set_coverage':((true[:,j]>=lo[:,j])&(true[:,j]<=hi[:,j])).mean(),'mean_grade_set_size':(hi[:,j]-lo[:,j]+1).mean()})
        for i,row in enumerate(test):
            for j,t in enumerate(targets): intervals.append({'repeat':repeat,'beer':int(data.meta.iloc[row].beer),'target':t,'continuous_prediction':pp[i,j],'lower':pp[i,j]-q[j],'upper':pp[i,j]+q[j],'relative_grade':point[i,j],'relative_grade_min':lo[i,j],'relative_grade_max':hi[i,j],'status':'fixed_training_relative_bins_not_absolute_raw_scores'})
        save_csv(pd.DataFrame({'repeat':repeat,'beer':data.meta.beer,'role':['train' if i in set(train) else 'calibration' if i in set(calibration) else 'test' for i in range(len(X))]}),dest/f'conformal_roles_{repeat}.csv')
    save_csv(pd.DataFrame(intervals),dest/'ASSUMPTION_ordinal_sets.csv'); save_csv(pd.DataFrame(coverage_rows),dest/'conformal_coverage.csv')
    write_json(dest/'status.json',{'complete':True,'outer_folds':15,'unique_beers':len(X),'grade_semantics':'relative_training_quantile_not_original_0_6_intensity','absolute_anchors_observed':False,'elapsed_seconds':time.perf_counter()-start,'model_originality':'standard_estimators_and_regularized_stacking_variant_no_first_invention_claim'})
    progress('S3 Q1 complete')
