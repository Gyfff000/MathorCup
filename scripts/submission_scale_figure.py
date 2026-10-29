"""Rebuild the scale figure using commensurate, auditable quantities."""
from pathlib import Path
import argparse,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import load_data,find_data
from src.common import save_csv,write_json,sha256

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir');a=ap.parse_args();D=load_data(find_data(a.data_dir));out=ROOT/'outputs/submission_review_20261029';rec=pd.read_csv(out/'field_sets.csv');exc=pd.read_csv(ROOT/'outputs/stage4/audit/chemical_source_exceptions.csv');fs=rec[(rec.audit_status=='all_comparable_pairs_reconciled')&(rec.best_transform=='log10_nonpositive_half_min')].field.tolist();assert len(fs)==37
 vals=[]
 for f in fs:
  x=D.X[f].to_numpy();c=D.calibrated[f].to_numpy();m=np.isfinite(x)&np.isfinite(c)&(c>0);vals.extend(zip(c[m],10**x[m]))
 xy=np.array(vals);counts=exc.field.value_counts();top=counts.head(8);other=int(counts.iloc[8:].sum());fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.3),constrained_layout=True);ax.scatter(xy[:,0],xy[:,1],s=3,alpha=.16,c='#287c91',rasterized=True);lo=max(min(xy.min(),1e-5),1e-8);hi=max(xy.max(),1e2);ax.plot([lo,hi],[lo,hi],color='#ac4d45',linestyle='--',linewidth=1);ax.set(xscale='log',yscale='log',xlabel='Reference concentration (reported units)',ylabel='Reconstructed concentration: $10^x$',title='(a) 37 log-scale fields, positive pairs');labels=list(top.index)+['Other 24 fields'];v=list(top.values)+[other];bp=bx.barh(range(len(v)),v,color=['#477f91']*8+['#999999']);bx.set_yticks(range(len(v)),labels,fontsize=7);bx.invert_yaxis();bx.set_xlabel('Number of mismatched cells');bx.set_title('(b) 80 exceptions in 32 fields');bx.bar_label(bp,padding=2,fontsize=8);bx.set_xlim(0,max(v)*1.18);fig.savefig(ROOT/'paper/submission_revision_20261029/figures/fig02_scale.png',dpi=220);plt.close(fig);save_csv(counts.rename_axis('field').reset_index(name='exception_count'),out/'exception_counts_by_field.csv');print('positive pairs',len(xy),'bar total',sum(v))
if __name__=='__main__':main()
