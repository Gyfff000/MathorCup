"""Publication figures from actual stage-three outputs; AI-assisted, 2026-10-25."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/stage3'; FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
def save(fig,name): fig.savefig(FIG/name,bbox_inches='tight'); plt.close(fig)

metrics=pd.read_csv(OUT/'q1/metrics_by_scope.csv'); names=['mean','ridge','extra_trees','gbr','fixed_half_blend','global_oof_blend','unshrunk_target_blend','shrunk_target_blend']
labels=['Mean','Ridge','ET','GBR','Half blend','Global blend','Target blend','Shrunk blend']
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
for ax,scope in zip(axes,['core30','all50']):
    a=metrics[metrics.scope==scope].set_index('model').loc[names]; ax.bar(np.arange(len(a)),a.mae,color=['#999999','#779ab2','#3978a8','#b36b73','#8ca98c','#87a7b7','#b79a75','#477e69']); ax.set_xticks(np.arange(len(a)),labels,rotation=40,ha='right'); ax.set_ylabel('MAE on supplied continuous targets'); ax.set_title('Core Aroma/Flavour: 30 targets' if scope=='core30' else 'All 50 targets')
save(fig,'prediction_by_scope.png')

coverage=pd.read_csv(OUT/'q1/conformal_coverage.csv'); fig,axes=plt.subplots(1,2,figsize=(10,3.2))
axes[0].hist(coverage.continuous_coverage,bins=np.linspace(.5,1,11),color='#3978a8'); axes[0].axvline(.9,color='black',ls='--'); axes[0].set_xlabel('Observed marginal continuous coverage'); axes[0].set_ylabel('Target-split combinations')
axes[1].hist(coverage.mean_grade_set_size,bins=10,color='#477e69'); axes[1].set_xlabel('Mean size of relative ordinal interval hull'); axes[1].set_ylabel('Target-split combinations'); fig.suptitle('Five train/calibration/test splits; relative grades are not raw intensity')
save(fig,'ordinal_uncertainty.png')

graph=pd.read_csv(OUT/'q2/graph_comparison.csv'); params=pd.read_csv(OUT/'q2/selected_parameters.csv'); fig,axes=plt.subplots(1,2,figsize=(10,3.3))
a=graph[graph.scope=='core30'].set_index('graph').loc[['none','real','shuffled']]; axes[0].bar(['No graph','Real graph','Shuffled graph'],a['mean'],color=['#999999','#3978a8','#b36b73']); axes[0].set_ylim(0,.45); axes[0].set_ylabel('Mean outer-fold core MAE')
for j,v in enumerate(a['mean']): axes[0].text(j,v+.006,f'{v:.6f}',ha='center',fontsize=9)
p=params[params.graph=='real']['lambda'].value_counts().sort_index(); axes[1].bar(p.index.astype(str),p.values,color='#477e69'); axes[1].set_xlabel('Graph penalty selected inside training'); axes[1].set_ylabel('Outer folds (15 total)'); fig.suptitle('Graph regularization ablation; no meaningful predictive gain observed')
save(fig,'graph_ablation.png')

top=pd.read_csv(OUT/'q2/top3_with_evidence.csv'); fig,ax=plt.subplots(figsize=(7,3.6)); good=top.evidence_status.str.startswith('comparatively'); ax.scatter(top.loc[~good,'bootstrap_frequency_conditional_on_full_selected_params'],top.loc[~good,'within_style_delta_on_selected_folds'],s=16,color='#84a9c2',label='Weak or incomplete'); ax.scatter(top.loc[good,'bootstrap_frequency_conditional_on_full_selected_params'],top.loc[good,'within_style_delta_on_selected_folds'],s=24,color='#b85e43',label='Current descriptive criteria'); ax.axhline(0,color='gray',lw=.7); ax.axvline(.5,color='gray',ls='--',lw=.7); ax.set_xlabel('Conditional bootstrap Top-3 frequency'); ax.set_ylabel('Within-style held-out permutation MAE change'); ax.legend(fontsize=8); save(fig,'factor_crossfit.png')

gen=pd.read_csv(OUT/'q3/generation_comparison.csv'); methods=['bootstrap','local_convex','constrained_local_convex','PCA_Gaussian']; styles=['Tripel','Kriek','Lambic']; colors=['#999999','#3978a8','#477e69','#b38a58']; fig,axes=plt.subplots(1,2,figsize=(11,3.5)); width=.19
for k,(method,color) in enumerate(zip(methods,colors)):
    a=gen[gen.method==method].groupby('style').mean(numeric_only=True).loc[styles]; x=np.arange(3)+(k-1.5)*width
    axes[0].bar(x,a.MMD2_fixed_training_bandwidth,width,color=color,label=method); axes[1].bar(x,a.independent_evaluator_target_prediction_fraction,width,color=color)
for ax in axes: ax.set_xticks(np.arange(3),styles)
axes[0].set_ylabel('MMD squared (shared training bandwidth)'); axes[1].set_ylabel('Independent evaluator predicts target style'); axes[1].set_ylim(0,1.05); axes[0].legend(fontsize=7); save(fig,'generation_independent.png')

transfer=pd.read_csv(OUT/'q3/transfer_independent_evaluation.csv'); fig,ax=plt.subplots(figsize=(7.5,3.6)); radii=sorted(transfer.radius.unique()); values=[]
for r in radii:
    a=transfer[transfer.radius==r]; t=a[a.source_role=='T_real_holdout']; values.append([(a.status!='no_feasible_nontrivial_candidate_source_retained').mean(),a.evaluator_proxy_acceptance.mean(),t.evaluator_proxy_acceptance.mean()])
values=np.array(values)
for j,(label,color) in enumerate([('Search model feasible (55 records)','#3978a8'),('Independent proxy acceptance (55 records)','#b36b73'),('Real-holdout source proxy acceptance (10 records)','#477e69')]): ax.plot(radii,values[:,j],marker='o',label=label,color=color)
ax.set_xlabel('Standardized change radius'); ax.set_ylabel('Fraction of records (repeated beers across splits)'); ax.set_ylim(-.03,1.03); ax.legend(fontsize=8,loc='upper left'); save(fig,'transfer_independent.png')

weights=pd.read_csv(OUT/'q1/blend_weights.csv'); fig,ax=plt.subplots(figsize=(7,3)); core=weights.target.str.startswith(('A_','F_')); ax.hist([weights.loc[core,'weight_GBR'],weights.loc[~core,'weight_GBR']],bins=np.linspace(0,1,11),label=['Core 30','Additional 20'],color=['#3978a8','#b38a58'],alpha=.8); ax.set_xlabel('GBR weight learned inside outer training only'); ax.set_ylabel('Target-fold combinations'); ax.legend(); save(fig,'blend_weights.png')
print('Saved seven stage-three figures')
