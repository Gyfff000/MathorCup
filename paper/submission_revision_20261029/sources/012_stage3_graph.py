"""Train-only graph ablations and cross-fitted exploratory explanations.
AI-assisted, OpenAI Codex, 2026-10-25. Graph regularization is an existing method.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import solve
from scipy.stats import rankdata
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from .common import preprocessing, progress, save_csv, write_json
from .q2 import signed_laplacian, top_indices
from .data import NON_MOLECULES
from .stage3_algorithms import paired_oof_bootstrap


def train_correlation(Z):
    with np.errstate(invalid='ignore',divide='ignore'):
        corr=np.corrcoef(rankdata(Z,axis=0),rowvar=False)
    return np.nan_to_num(corr)


def solve_coeff(Z,Y,L,alpha,lam):
    return solve(Z.T@Z+alpha*np.eye(Z.shape[1])+lam*L,Z.T@(Y-Y.mean(0)),assume_a='pos',check_finite=False)


def candidates(cfg,kind):
    return [(alpha,lam,tau) for alpha in cfg['ridge_alphas'] for lam in ([0] if kind=='none' else cfg['graph_lambdas']) for tau in ([.7] if lam==0 else cfg['graph_thresholds'])]


def fit_tuned_graph(X,Y,cfg,kind,core,seed):
    choices=candidates(cfg,kind); scores=np.zeros(len(choices))
    for fold,(train,test) in enumerate(KFold(3,shuffle=True,random_state=seed).split(X)):
        prep=preprocessing(); Z=prep.fit_transform(X[train]); V=prep.transform(X[test]); corr=train_correlation(Z)
        lhs=Z.T@Z; rhs=Z.T@(Y[train]-Y[train].mean(0)); identity=np.eye(Z.shape[1])
        layers={tau:signed_laplacian(corr,tau) for tau in cfg['graph_thresholds']}
        perms=[np.random.default_rng(seed+1009*k+fold).permutation(Z.shape[1]) for k in range(cfg['graph_null_repeats'])] if kind=='null' else [None]
        for c,(alpha,lam,tau) in enumerate(choices):
            L=layers[tau] if lam else np.zeros_like(lhs)
            for perm in perms:
                LL=L[np.ix_(perm,perm)] if perm is not None else L
                beta=solve(lhs+alpha*identity+lam*LL,rhs,assume_a='pos',check_finite=False)
                pp=V@beta+Y[train].mean(0)
                scores[c]+=np.abs(Y[test][:,core]-pp[:,core]).sum()/(len(Y)*len(core)*len(perms))
    best=min(range(len(choices)),key=lambda i:(scores[i],choices[i][1],-choices[i][0]))
    alpha,lam,tau=choices[best]; prep=preprocessing(); Z=prep.fit_transform(X); corr=train_correlation(Z); L=signed_laplacian(corr,tau)
    if kind=='none': L=np.zeros_like(L)
    betas=[]
    for k in range(cfg['graph_null_repeats'] if kind=='null' else 1):
        perm=np.random.default_rng(seed+20000+k).permutation(Z.shape[1]) if kind=='null' else None
        LL=L[np.ix_(perm,perm)] if perm is not None else L
        betas.append(solve_coeff(Z,Y,LL,alpha,lam))
    return {'prep':prep,'Z':Z,'corr':corr,'beta':betas,'mean':Y.mean(0),'params':{'alpha':alpha,'lambda':lam,'tau':tau},'inner_core_mae':float(scores[best])}


def within_style_indices(labels,rng):
    labels=np.asarray(labels); indices=np.arange(len(labels))
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label); indices[ids]=rng.permutation(ids)
    return indices


def run_graph(data,out,cfg):
    started=time.perf_counter(); dest=Path(out)/'q2'; dest.mkdir(parents=True,exist_ok=True)
    X,Y=data.X.to_numpy(),data.Y.to_numpy(); features,targets=list(data.X),list(data.Y); labels=data.meta.tasting_category_fine.to_numpy()
    core=np.array([j for j,t in enumerate(targets) if t.startswith(('A_','F_'))]); eligible=np.array([j for j,t in enumerate(features) if t not in NON_MOLECULES])
    counts=np.zeros(len(Y)); totals={kind:np.zeros_like(Y) for kind in ['none','real']}; rows=[]; chosen=[]; effects=[]; modules=[]; style_effects=[]; selected_sets=[]
    for fold,(train,test) in enumerate(RepeatedStratifiedKFold(n_splits=3,n_repeats=cfg['outer_repeats'],random_state=cfg['seed']).split(X,labels)):
        progress(f'S3 Q2 graph fold {fold+1}/15, elapsed {(time.perf_counter()-started)/60:.1f} min')
        fits={kind:fit_tuned_graph(X[train],Y[train],cfg,kind,core,cfg['seed']+fold) for kind in ['none','real','null']}
        for kind,fitted in fits.items():
            V=fitted['prep'].transform(X[test])
            for k,beta in enumerate(fitted['beta']):
                pred=V@beta+fitted['mean']
                for scope,ix in [('core30',core),('all50',np.arange(Y.shape[1]))]:
                    rows.append({'fold':fold,'repeat':fold//3,'graph':'shuffled' if kind=='null' else kind,'null_rep':k if kind=='null' else -1,'scope':scope,'mae':np.abs(Y[test][:,ix]-pred[:,ix]).mean()})
                if kind in totals: totals[kind][test]+=pred
            chosen.append({'fold':fold,'graph':'shuffled' if kind=='null' else kind,**fitted['params'],'inner_core_mae':fitted['inner_core_mae']})
        counts[test]+=1
        fit=fits['real']; Z=fit['Z']; V=fit['prep'].transform(X[test]); B=fit['beta'][0]; original=V@B+fit['mean']; base=np.abs(Y[test]-original).mean(0)
        rng=np.random.default_rng(cfg['seed']+fold)
        fold_chosen=np.stack([top_indices(B[:,k],eligible) for k in range(Y.shape[1])]); selected_sets.append(fold_chosen)
        for k,target in enumerate(targets):
            for rank,j in enumerate(fold_chosen[k],1):
                all_delta=[]; within_delta=[]
                for repeat in range(cfg['permutation_repeats']):
                    ix=rng.permutation(len(V)); ixw=within_style_indices(labels[test],rng)
                    all_delta.append(np.abs(Y[test,k]-(original[:,k]+(V[ix,j]-V[:,j])*B[j,k])).mean()-base[k])
                    within_delta.append(np.abs(Y[test,k]-(original[:,k]+(V[ixw,j]-V[:,j])*B[j,k])).mean()-base[k])
                effects.append({'fold':fold,'target':target,'rank':rank,'field':features[j],'coefficient':B[j,k],
                    'heldout_delta_mae':np.mean(all_delta),'within_style_delta_mae':np.mean(within_delta),
                    'singleton_test_style_fraction':np.mean([np.sum(labels[test]==label)==1 for label in labels[test]])})
        # Joint module perturbations preserve internal feature relationships.
        graph=nx.Graph(); graph.add_nodes_from(range(X.shape[1])); W=np.abs(fit['corr'])>=fit['params']['tau']; np.fill_diagonal(W,False)
        graph.add_edges_from(zip(*np.where(np.triu(W,1))))
        interesting={frozenset(nx.node_connected_component(graph,int(j))) for j in np.unique(fold_chosen)}
        for group in interesting:
            ix=np.array(sorted(group)); keep=np.setdiff1d(np.arange(X.shape[1]),ix)
            L=signed_laplacian(fit['corr'][np.ix_(keep,keep)],fit['params']['tau'])
            reduced=solve_coeff(Z[:,keep],Y[train],L,fit['params']['alpha'],fit['params']['lambda'])
            drop=V[:,keep]@reduced+fit['mean']; joint=np.zeros(Y.shape[1])
            for repeat in range(cfg['permutation_repeats']):
                order=within_style_indices(labels[test],rng)
                pp=original+(V[np.ix_(order,ix)]-V[:,ix])@B[ix]
                joint+=np.abs(Y[test]-pp).mean(0)-base
            for k,t in enumerate(targets):
                if set(fold_chosen[k])&group:
                    modules.append({'fold':fold,'target':t,'module_fields':';'.join(features[j] for j in ix),'module_n':len(ix),'heldout_module_drop_mae':np.abs(Y[test,k]-drop[:,k]).mean()-base[k],'within_style_joint_permutation_mae':joint[k]/cfg['permutation_repeats']})
        # Style-centering is an explanatory diagnostic, not a Q1 model input.
        zr=Z.copy(); yr=Y[train].copy()
        for label in np.unique(labels[train]):
            ids=np.flatnonzero(labels[train]==label); zr[ids]-=Z[ids].mean(0); yr[ids]-=Y[train][ids].mean(0)
        adjusted=solve_coeff(zr,yr,signed_laplacian(train_correlation(zr),fit['params']['tau']),fit['params']['alpha'],fit['params']['lambda'])
        for k,t in enumerate(targets):
            adjusted_set=set(top_indices(adjusted[:,k],eligible))
            for j in fold_chosen[k]: style_effects.append({'fold':fold,'target':t,'field':features[j],'still_top3_after_style_centering':j in adjusted_set,'style_adjusted_coefficient':adjusted[j,k]})
        save_csv(pd.DataFrame(rows),dest/'graph_fold_metrics.csv'); save_csv(pd.DataFrame(chosen),dest/'selected_parameters.csv')
    for kind,pred in totals.items(): save_csv(pd.concat([data.meta[['beer']],pd.DataFrame(pred/counts[:,None],columns=targets)],axis=1),dest/f'oof_graph_{kind}.csv')
    summary=pd.DataFrame(rows).groupby(['graph','scope']).mae.agg(['mean','std']).reset_index(); save_csv(summary,dest/'graph_comparison.csv')
    save_csv(pd.DataFrame(effects),dest/'crossfit_top3_effects.csv'); save_csv(pd.DataFrame(modules),dest/'crossfit_module_effects.csv'); save_csv(pd.DataFrame(style_effects),dest/'style_centering_diagnostics.csv')
    paired=[]
    for scope,ix in [('core30',core),('all50',np.arange(Y.shape[1]))]:
        r=paired_oof_bootstrap(Y[:,ix],totals['real'][:,ix]/counts[:,None],totals['none'][:,ix]/counts[:,None],labels); r['target_scope']=scope; paired.append(r)
    write_json(dest/'paired_graph_vs_none.json',paired)
    progress('S3 Q2: full descriptive graph and 200 conditional bootstrap replicates')
    full=fit_tuned_graph(X,Y,cfg,'real',core,cfg['seed']+500); beta=full['beta'][0]; frequency=np.zeros_like(beta); rng=np.random.default_rng(cfg['seed']); p=full['params']
    for b in range(cfg['bootstrap_repeats']):
        ix=rng.integers(0,len(X),len(X)); z=preprocessing().fit_transform(X[ix]); L=signed_laplacian(train_correlation(z),p['tau']); bb=solve_coeff(z,Y[ix],L,p['alpha'],p['lambda'])
        for k in range(Y.shape[1]): frequency[top_indices(bb[:,k],eligible),k]+=1
    frequency/=cfg['bootstrap_repeats']; effect=pd.DataFrame(effects); adjusted=pd.DataFrame(style_effects); meta=data.chemical_meta.set_index('field'); top=[]
    for k,t in enumerate(targets):
        for rank,j in enumerate(top_indices(beta[:,k],eligible),1):
            selected=effect[(effect.target==t)&(effect.field==features[j])]; centered=adjusted[(adjusted.target==t)&(adjusted.field==features[j])]
            within=float(selected.within_style_delta_mae.mean()) if len(selected) else None
            delta=float(selected.heldout_delta_mae.mean()) if len(selected) else None
            supported=frequency[j,k]>=.5 and len(selected)>=5 and delta is not None and delta>0 and within is not None and within>0
            top.append({'target':t,'rank':rank,'field':features[j],'coefficient':beta[j,k],'bootstrap_frequency_conditional_on_full_selected_params':frequency[j,k],
                'crossfit_selection_count':len(selected),'heldout_delta_on_selected_folds':delta,'within_style_delta_on_selected_folds':within,
                'style_centering_retention_fraction':float(centered.still_top3_after_style_centering.mean()) if len(centered) else None,
                'origin':meta.at[features[j],'origin_paper'],'unit':meta.at[features[j],'unit'],'attachment_off_flavors':meta.at[features[j],'off_flavors'],
                'evidence_status':'comparatively_supported_exploratory_association' if supported else 'weak_or_incomplete_evidence',
                'chemical_review':'target_specific_literature_not_yet_complete','causal_status':'not_identified'})
    save_csv(pd.DataFrame(top),dest/'top3_with_evidence.csv'); save_csv(pd.DataFrame(frequency,index=features,columns=targets).reset_index(names='field'),dest/'bootstrap_frequencies.csv')
    write_json(dest/'status.json',{'complete':True,'outer_folds':15,'null_graphs_per_fit':cfg['graph_null_repeats'],'full_graph_params':p,'bootstrap_repeats':cfg['bootstrap_repeats'],'bootstrap_params_reselected':False,
        'bootstrap_caveat':'conditional_on_hyperparameters_selected_on_full_data; descriptive_stability_not_independent_inference','supported_top3':sum(r['evidence_status'].startswith('comparatively') for r in top),'scope':'exploratory_association_not_causal','elapsed_seconds':time.perf_counter()-started})
    progress('S3 Q2 complete')
