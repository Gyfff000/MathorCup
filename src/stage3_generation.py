"""Disjoint-search/evaluation conditional generation and Pareto transfer.
AI-assisted reference implementation; no actual brewing or tasting validation.
"""
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from .common import preprocessing, save_csv, write_json, progress
from .data import AGGREGATES, CHEMICAL_DEFECTS
from .q3 import ConditionalPCAGaussian
from .stage2 import local_convex_samples
from .stage3_algorithms import pareto_minimize, threshold_excess


def mmd_fixed_bandwidth(a,b,bandwidth):
    if len(a)<2 or len(b)<2: return np.nan
    return float(np.exp(-cdist(a,a,'sqeuclidean')/(2*bandwidth)).mean()+np.exp(-cdist(b,b,'sqeuclidean')/(2*bandwidth)).mean()-2*np.exp(-cdist(a,b,'sqeuclidean')/(2*bandwidth)).mean())


def loo_support_radius(z):
    if len(z)<2: return np.nan
    distance=cdist(z,z); np.fill_diagonal(distance,np.inf)
    return float(np.quantile(distance.min(1),.95))


def constrained_convex(real,generator,style_positions,n,rng):
    Z=generator.prep.transform(real); distance=cdist(Z,Z); np.fill_diagonal(distance,np.inf)
    neighbors=np.argsort(distance,axis=1)[:,:min(3,len(real)-1)]; radius=loo_support_radius(Z); result=[]; attempts=0
    if len(real)<2: return real[np.zeros(n,dtype=int)].copy(),{'attempts':n,'singleton_fallback':True,'support_radius':None}
    while len(result)<n and attempts<n*250:
        attempts+=1; a=rng.integers(len(real)); b=rng.choice(neighbors[a]); weight=rng.uniform(.1,.9)
        value=weight*real[a]+(1-weight)*real[b]
        for j in style_positions:
            value[j]=np.log10(weight*10**real[a,j]+(1-weight)*10**real[b,j])
        value=np.clip(value,generator.low,generator.high)
        if cdist(generator.prep.transform(value[None]),Z).min()<=radius+1e-9: result.append(value)
    return np.asarray(result).reshape((-1,real.shape[1])),{'attempts':attempts,'singleton_fallback':False,'support_radius':radius}


def locked_transfer_candidates(X,labels,beers,G,generator,searcher,sensory_a,consensus_a,features,targets,cfg):
    positions=[features.index(t) for t in CHEMICAL_DEFECTS]; logpos=[features.index(t) for t in ['DMS','X4VG']]
    target_class=list(searcher.classes_).index('Tripel'); stale=targets.index('stale_hops')
    target_data=generator.prep.named_steps['scale'].inverse_transform(generator.prep.transform(X[G[labels[G]=='Tripel']]))
    target_Z=generator.prep.transform(target_data); support=loo_support_radius(target_Z)
    result=[]
    for i in np.flatnonzero(labels=='Pils/Lager'):
        source=np.where(np.isfinite(X[i]),X[i],generator.median); sourceZ=generator.prep.transform(source[None])
        source_p=searcher.predict_proba(source[None])[0,target_class]; stale_a=sensory_a.predict(source[None])[0,stale]; stale_c=consensus_a.predict(source[None])[0,stale]
        cleaned=np.clip(source,generator.low,generator.high); cleaned[positions]=np.minimum(cleaned[positions],source[positions]); clean_p=searcher.predict_proba(cleaned[None])[0,target_class]
        values=[]
        for anchor in target_data:
            for weight in np.linspace(.025,1,cfg['transfer_weights']):
                v=np.clip(source+weight*(anchor-source),generator.low,generator.high); v[positions]=np.minimum(v[positions],source[positions]); values.append(v)
        values=np.asarray(values); Z=generator.prep.transform(values); changes=np.linalg.norm(Z-sourceZ,axis=1)/np.sqrt(len(features)); support_distance=cdist(Z,target_Z).min(1)
        probabilities=searcher.predict_proba(values)[:,target_class]; pa=sensory_a.predict(values); pc=consensus_a.predict(values)
        bounds=((values>=generator.low-1e-9)&(values<=generator.high+1e-9)).all(1)
        nonincrease=(values[:,positions]<=source[positions]+1e-9).all(1)
        known_proxy_risk=np.maximum(threshold_excess(10**values[:,logpos[0]],19.),threshold_excess(10**values[:,logpos[1]],.2))
        valid=bounds&nonincrease&(pa[:,stale]<=stale_a+1e-9)&(pc[:,stale]<=stale_c+1e-9)&(probabilities>=max(cfg['transfer_target_probability_floor'],source_p+.01,clean_p+.01))&(support_distance<=support+1e-9)&(changes>1e-8)
        for radius in cfg['trust_radii']:
            ids=np.flatnonzero(valid&(changes<=radius)); pareto_ids=[]
            if len(ids):
                nd=pareto_minimize(np.c_[changes[ids],support_distance[ids],known_proxy_risk[ids]]); pareto_ids=ids[nd]; j=int(pareto_ids[np.argmin(changes[pareto_ids])]); value=values[j]; status='model_feasible_nontrivial' if np.isfinite(X[i,positions]).all() else 'conditional_on_imputed_source_defect'
                p=float(probabilities[j]); change=float(changes[j]); target_distance=float(support_distance[j])
            else:
                value=source.copy(); status='no_feasible_nontrivial_candidate_source_retained'; p=float(source_p); change=0.; target_distance=float(cdist(sourceZ,target_Z).min())
            result.append({'row':int(i),'beer':int(beers[i]),'radius':radius,'source':source.copy(),'candidate':value.copy(),'status':status,'n_candidates':len(values),'n_valid':len(ids),'n_pareto':len(pareto_ids),
                'search_source_probability':float(source_p),'search_candidate_probability':p,'standardized_change':change,'target_support_distance':target_distance,
                'all_provided_source_defects_observed':bool(np.isfinite(X[i,positions]).all()),'missing_defects':','.join(features[j] for j in positions if not np.isfinite(X[i,j]))})
    return result


def run_generation(data,out,cfg):
    start=time.perf_counter(); dest=Path(out)/'q3'; dest.mkdir(parents=True,exist_ok=True)
    features=[f for f in data.X if f not in AGGREGATES]; targets=list(data.Y); X=data.X[features].to_numpy(); Y=data.Y.to_numpy(); labels=data.meta.tasting_category_fine.to_numpy(); beers=data.meta.beer.to_numpy()
    core=[j for j,t in enumerate(targets) if t.startswith(('A_','F_'))]; log_positions=[features.index(f) for f in ['DMS','X4VG']]; stale=targets.index('stale_hops'); positions=[features.index(f) for f in CHEMICAL_DEFECTS]
    metrics=[]; transfers=[]; checks=[]; roles=[]; evaluator_rows=[]; locked=[]; profiles=[]; portraits=[]; generated_profiles=[]
    for repeat in range(cfg['generation_repeats']):
        seed=cfg['seed']+repeat; rng=np.random.default_rng(seed); progress(f'S3 Q3 disjoint partition {repeat+1}/5')
        dev,T=train_test_split(np.arange(len(X)),test_size=50,stratify=labels,random_state=seed)
        G,J=train_test_split(dev,test_size=100,stratify=labels[dev],random_state=seed+999)
        assert len(G)==len(J)==100 and len(T)==50 and not(set(G)&set(J) or set(G)&set(T) or set(J)&set(T))
        roles.extend({'repeat':repeat,'beer':int(beers[i]),'role':'G_search_training' if i in set(G) else 'J_evaluator_training' if i in set(J) else 'T_real_holdout'} for i in range(len(X)))
        generator=ConditionalPCAGaussian(10,seed).fit(X[G],labels[G])
        searcher=Pipeline([('prep',preprocessing()),('model',LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=seed))]).fit(X[G],labels[G])
        sensory_a=Pipeline([('prep',preprocessing()),('model',Ridge(alpha=100))]).fit(X[G],Y[G])
        consensus_a=Pipeline([('prep',preprocessing()),('model',ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=.8,random_state=seed,n_jobs=2))]).fit(X[G],Y[G])
        generated=[]
        for style in cfg['generation_styles']:
            real=generator.prep.named_steps['scale'].inverse_transform(generator.prep.transform(X[G[labels[G]==style]]))
            constrained,info=constrained_convex(real,generator,log_positions,cfg['generation_n'],rng)
            samples={'bootstrap':real[rng.integers(0,len(real),cfg['generation_n'])],'local_convex':local_convex_samples(real,cfg['generation_n'],rng),
                'PCA_Gaussian':generator.sample(style,cfg['generation_n'],rng),'constrained_local_convex':constrained}
            for method,values in samples.items(): generated.append({'style':style,'method':method,'values':values,'training_n':len(real),'extra':info if method=='constrained_local_convex' else {}})
        selected=locked_transfer_candidates(X,labels,beers,G,generator,searcher,sensory_a,consensus_a,features,targets,cfg)
        digest=hashlib.sha256()
        for group in generated: digest.update(group['values'].tobytes())
        for row in selected: digest.update(row['candidate'].tobytes())
        lock={'repeat':repeat,'candidate_sha256':digest.hexdigest(),'selection_used_J_or_T':False,'n_generated':sum(len(a['values']) for a in generated),'n_transfer_records':len(selected)}
        locked.append(lock); write_json(dest/f'locked_candidates_{repeat}.json',lock)
        # J models are fitted only after candidates and their order are fixed.
        evaluator=Pipeline([('prep',preprocessing()),('model',ExtraTreesClassifier(n_estimators=200,min_samples_leaf=2,max_features=.7,class_weight='balanced',random_state=seed+3000,n_jobs=2))]).fit(X[J],labels[J])
        sensory_b=Pipeline([('prep',preprocessing()),('model',ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=.8,random_state=seed+4000,n_jobs=2))]).fit(X[J],Y[J])
        acc_a=balanced_accuracy_score(labels[T],searcher.predict(X[T])); acc_b=balanced_accuracy_score(labels[T],evaluator.predict(X[T])); y_a=sensory_a.predict(X[T]); y_b=sensory_b.predict(X[T])
        evaluator_rows.append({'repeat':repeat,'searcher_holdout_balanced_accuracy':acc_a,'evaluator_holdout_balanced_accuracy':acc_b,'search_sensory_core30_mae':np.abs(Y[T][:,core]-y_a[:,core]).mean(),'evaluator_sensory_core30_mae':np.abs(Y[T][:,core]-y_b[:,core]).mean()})
        allG=generator.prep.transform(X[G]); distance=cdist(allG,allG,'sqeuclidean'); bandwidth=float(np.median(distance[distance>0]))
        for group in generated:
            style,method,values=group['style'],group['method'],group['values']; test=X[T[labels[T]==style]]; Z=generator.prep.transform(values) if len(values) else np.empty((0,len(features)))
            if not len(values):
                metrics.append({'repeat':repeat,'style':style,'method':method,'n_generated':0,'n_G_style':group['training_n'],'n_T_style':len(test),'status':'generation_budget_exhausted_no_sample'}); continue
            testZ=generator.prep.transform(test); nn=cdist(Z,allG).min(1)
            prob_a=searcher.predict_proba(values)[:,list(searcher.classes_).index(style)]; prob_b=evaluator.predict_proba(values)[:,list(evaluator.classes_).index(style)]
            jstyle=X[J[labels[J]==style]]; jz=evaluator.named_steps['prep'].transform(jstyle); threshold=loo_support_radius(jz); distances_b=cdist(evaluator.named_steps['prep'].transform(values),jz).min(1)
            ks=[ks_2samp(values[:,j],test[:,j][np.isfinite(test[:,j])]).statistic for j in range(X.shape[1]) if np.isfinite(test[:,j]).any()]
            metrics.append({'repeat':repeat,'style':style,'method':method,'n_generated':len(values),'n_G_style':group['training_n'],'n_J_style':len(jstyle),'n_T_style':len(test),
                'MMD2_fixed_training_bandwidth':mmd_fixed_bandwidth(Z,testZ,bandwidth),'KS_mean':np.mean(ks),'exact_copy_fraction':(nn<1e-8).mean(),
                'search_probability':prob_a.mean(),'independent_evaluator_probability':prob_b.mean(),'independent_evaluator_target_prediction_fraction':np.mean(evaluator.predict(values)==style),
                'J_support_fraction':np.mean(distances_b<=threshold) if np.isfinite(threshold) else np.nan,
                'provided_bounds_violation_fraction':np.mean(~((values>=generator.low-1e-9)&(values<=generator.high+1e-9)).all(1)),
                'sensory_proxy_disagreement':np.abs(sensory_a.predict(values)-sensory_b.predict(values)).mean(),
                'bandwidth_from_G':bandwidth,'status':'statistical_profiles_not_actual_tasting','singleton_fallback':group['extra'].get('singleton_fallback',False)})
            if repeat==0:
                ids=[f'S3_{method}_{style}_{i+1:03d}' for i in range(len(values))]
                frame=pd.DataFrame(values,columns=features); frame.insert(0,'style',style); frame.insert(0,'method',method); frame.insert(0,'generated_id',ids); generated_profiles.append(frame)
                if method=='constrained_local_convex':
                    portrait=pd.DataFrame(sensory_a.predict(values),columns=targets); portrait.insert(0,'style',style); portrait.insert(0,'generated_id',ids); portraits.append(portrait)
        for row in selected:
            i=row['row']; source,candidate=row['source'],row['candidate']; pb0=evaluator.predict_proba(source[None])[0,list(evaluator.classes_).index('Tripel')]; pb1=evaluator.predict_proba(candidate[None])[0,list(evaluator.classes_).index('Tripel')]
            sb0=sensory_b.predict(source[None])[0,stale]; sb1=sensory_b.predict(candidate[None])[0,stale]
            nontrivial=row['status']!='no_feasible_nontrivial_candidate_source_retained'
            b_accept=nontrivial and pb1>pb0+.01 and evaluator.predict(candidate[None])[0]=='Tripel' and sb1<=sb0+1e-9
            record={k:v for k,v in row.items() if k not in ['row','source','candidate']}
            record.update(repeat=repeat,source_role='G_search_training' if i in set(G) else 'J_evaluator_training' if i in set(J) else 'T_real_holdout',
                evaluator_source_probability=pb0,evaluator_candidate_probability=pb1,evaluator_predicts_Tripel=bool(evaluator.predict(candidate[None])[0]=='Tripel'),
                evaluator_stale_source=sb0,evaluator_stale_candidate=sb1,evaluator_proxy_acceptance=bool(b_accept),absolute_chemical_certification=False,real_tasting_validated=False)
            transfers.append(record)
            profiles.append({'repeat':repeat,'beer':int(beers[i]),'radius':row['radius'],'status':row['status'],**dict(zip(features,candidate))})
            for field,j in zip(CHEMICAL_DEFECTS,positions):
                raw=float(data.calibrated.iloc[i][field]) if field in data.calibrated else np.nan
                supported=bool(np.isfinite(raw) and raw>0 and np.isclose(X[i,j],np.log10(raw) if field in ['DMS','X4VG'] else raw,rtol=1e-6,atol=1e-7))
                checks.append({'repeat':repeat,'beer':int(beers[i]),'radius':row['radius'],'field':field,'source_provided':X[i,j],'source_for_search':source[j],'candidate_provided':candidate[j],
                    'provided_nonincrease':bool(candidate[j]<=source[j]+1e-9),'source_provided_observed':bool(np.isfinite(X[i,j])),
                    'source_positive_calibrated_pair_matches':supported,'absolute_threshold_available_in_attachment':field!='X.E..2.nonenal',
                    'complete_absolute_certification':False,'caveat':'scale_and_detection_limit_conditions_not_fully_verified; candidate_has_no_new_measurement'})
        save_csv(pd.DataFrame(metrics),dest/'generation_comparison.csv'); save_csv(pd.DataFrame(transfers),dest/'transfer_independent_evaluation.csv')
    save_csv(pd.DataFrame(roles),dest/'disjoint_roles.csv'); save_csv(pd.DataFrame(evaluator_rows),dest/'real_holdout_diagnostics.csv'); save_csv(pd.DataFrame(checks),dest/'transfer_defect_checks.csv')
    save_csv(pd.DataFrame(profiles),dest/'transfer_profiles.csv'); save_csv(pd.concat(generated_profiles,ignore_index=True),dest/'generated_profiles_first_split.csv'); save_csv(pd.concat(portraits,ignore_index=True),dest/'generated_sensory_proxy_first_split.csv')
    summary=pd.DataFrame(metrics).groupby(['method','style'])[['MMD2_fixed_training_bandwidth','KS_mean','exact_copy_fraction','search_probability','independent_evaluator_probability','independent_evaluator_target_prediction_fraction','provided_bounds_violation_fraction']].mean().reset_index(); save_csv(summary,dest/'generation_summary.csv')
    write_json(dest/'status.json',{'complete':True,'repeats':5,'G_n':100,'J_n':100,'T_n':50,'basic_fields':len(features),'excluded_unverified_aggregates':sorted(AGGREGATES),
        'candidate_locks':locked,'independent_evaluation_scope':'J is disjoint within the current run; all original beers were seen in earlier research stages, not new external validation',
        'absolute_chemical_certification':False,'real_brewing_or_tasting':False,'elapsed_seconds':time.perf_counter()-start})
    progress('S3 Q3 complete')
