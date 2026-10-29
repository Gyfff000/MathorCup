"""Descriptive sample matrices and Spearman heatmaps for the paper."""
from pathlib import Path
import argparse,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import load_data,find_data
from src.common import preprocessing,save_csv,write_json,sha256

def draw(matrix,xlabels,ylabels,path):
 fig,ax=plt.subplots(figsize=(8.8,6.3),constrained_layout=True);im=ax.imshow(matrix.to_numpy(),vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto');ax.set_xticks(range(len(xlabels)),xlabels,rotation=40,ha='right',fontsize=9);ax.set_yticks(range(len(ylabels)),ylabels,fontsize=9)
 for i in range(len(ylabels)):
  for j in range(len(xlabels)):
   v=matrix.iloc[i,j];ax.text(j,i,f'{0. if abs(v)<.005 else v:.2f}' if np.isfinite(v) else 'NA',ha='center',va='center',fontsize=8,color='white' if np.isfinite(v) and abs(v)>.55 else 'black')
 ax.set_xticks(np.arange(-.5,len(xlabels),1),minor=True);ax.set_yticks(np.arange(-.5,len(ylabels),1),minor=True);ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='minor',bottom=False,left=False);cb=fig.colorbar(im,ax=ax,shrink=.82,pad=.025);cb.set_label('Spearman correlation',fontsize=9);fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir');a=ap.parse_args();d=load_data(find_data(a.data_dir));beers=d.meta.beer.to_numpy();targets=list(d.Y);Y=pd.read_csv(ROOT/'outputs/stage4/audit/observed_panel_means_0_6.csv').set_index('beer').loc[beers,targets]
 dest=ROOT/'outputs/matrix_figures_20261029';dest.mkdir(parents=True,exist_ok=True);figdir=ROOT/'paper/latex_20261028/figures';fields=['acetaldehyde','DMS','ethyl_acetate','ethyl_hexanoate','X4VG','iron','X2.3.butanedione','X.E..2.nonenal'];sens=['A_esters_all','A_hops_all','F_malt_all','X4vg','diacetyl','dms','metallic','stale_hops']; labels=['Acetaldehyde','DMS','Ethyl acetate','Ethyl hexanoate','4VG','Iron','Diacetyl','trans-2-Nonenal']; yl=['Ester aroma','Hop aroma','Malt flavour','4VG','Diacetyl','DMS','Metallic','Stale hops']
 X=d.X; full=pd.concat([X.reset_index(drop=True),Y.reset_index(drop=True)],axis=1); cross_full=full.corr(method='spearman').loc[targets,list(X)]; save_csv(cross_full.reset_index(names='target'),dest/'chemical_sensory_spearman_50x231.csv'); full_counts=Y.notna().to_numpy(dtype=int).T@X.notna().to_numpy(dtype=int); save_csv(pd.DataFrame(full_counts,index=targets,columns=X.columns).reset_index(names='target'),dest/'chemical_sensory_pair_counts_50x231.csv'); corr=X.corr(method='spearman');save_csv(corr.reset_index(names='field'),dest/'chemical_spearman_231.csv');sub=corr.loc[fields,fields];save_csv(sub.reset_index(names='field'),dest/'chemical_spearman_subset.csv');draw(sub,labels,labels,figdir/'chemical_matrix.png')
 zz=pd.concat([X[fields].reset_index(drop=True),Y[sens].reset_index(drop=True)],axis=1); cross=zz.corr(method='spearman').loc[sens,fields];counts=pd.DataFrame([[int((X[f].notna().to_numpy() & Y[t].notna().to_numpy()).sum()) for f in fields] for t in sens],index=sens,columns=fields);save_csv(cross.reset_index(names='target'),dest/'chemical_sensory_spearman.csv');save_csv(counts.reset_index(names='target'),dest/'chemical_sensory_pair_counts.csv');draw(cross,labels,yl,figdir/'cross_matrix.png')
 Z=preprocessing().fit_transform(X);ix=list(X).index; mini=pd.DataFrame(Z[:5,[ix(f) for f in fields[:6]]],columns=fields[:6]);mini.insert(0,'beer',beers[:5]);save_csv(mini,dest/'descriptive_standardized_example.csv');sample=Y.iloc[:5][sens[:4]].copy();sample.insert(0,'beer',beers[:5]);save_csv(sample,dest/'panel_mean_example.csv')
 write_json(dest/'manifest.json',{'scope':'full historical dataset descriptive Spearman pairwise complete; no supervised models trained; sample standardization fitted to all historical beers for display only','chemical_n':len(X),'cross_pair_n_min':int(counts.to_numpy().min()),'cross_pair_n_max':int(counts.to_numpy().max()),'source_panel_sha256':sha256(ROOT/'outputs/stage4/audit/observed_panel_means_0_6.csv'),'source_code_sha256':sha256(Path(__file__))});print('matrix counts',counts.to_numpy().min(),counts.to_numpy().max());print('X example');print(mini.round(3).to_string(index=False));print('Y example');print(sample.round(3).to_string(index=False));print('cross');print(cross.round(3).to_string())
if __name__=='__main__':main()
