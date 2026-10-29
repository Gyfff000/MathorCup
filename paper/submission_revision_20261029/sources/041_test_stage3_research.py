"""Check actual isolation and mathematical invariants of the executed study."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from src.stage3_prediction import learn_blend_nested
from src.stage3_graph import fit_tuned_graph, within_style_indices
from src.stage3_generation import mmd_fixed_bandwidth, constrained_convex
from src.q3 import ConditionalPCAGaussian

ROOT=Path(__file__).resolve().parents[1]


def test_nested_blend_base_predictions_never_predict_their_own_training_rows(monkeypatch):
    import src.stage3_prediction as module
    calls=[]
    def fit(X,Y,cfg): return set(X[:,0].astype(int))
    def predict(seen,X):
        assert not seen.intersection(X[:,0].astype(int)); calls.append((seen,set(X[:,0].astype(int))))
        return np.zeros((len(X),2)),np.ones((len(X),2))
    monkeypatch.setattr(module,'fit_pair',fit); monkeypatch.setattr(module,'predict_pair',predict)
    X=np.c_[np.arange(12),np.arange(12)**2]; Y=np.c_[np.arange(12)/12,np.arange(12)/24]
    w,_,_,eta,_=learn_blend_nested(X,Y,{'seed':1,'blend_penalties':[0,.1]},np.array([0]),0)
    assert len(calls)==12 and np.all((w>=0)&(w<=1)) and eta in [0,.1]


def test_zero_penalty_graph_and_permuted_graph_reduce_to_same_ridge():
    rng=np.random.default_rng(5); X=rng.normal(size=(18,4)); Y=rng.normal(size=(18,2))
    cfg={'ridge_alphas':[10],'graph_lambdas':[0],'graph_thresholds':[.7],'graph_null_repeats':2}
    fits=[fit_tuned_graph(X,Y,cfg,k,np.array([0]),17) for k in ['none','real','null']]
    for f in fits:
        for b in f['beta']: np.testing.assert_allclose(b,fits[0]['beta'][0],atol=1e-12)


def test_style_permutation_preserves_each_style_and_singleton():
    labels=np.array(['A','A','B','B','B','C']); order=within_style_indices(labels,np.random.default_rng(9))
    np.testing.assert_array_equal(labels[order],labels); assert sorted(order)==list(range(6)) and order[-1]==5


def test_fixed_bandwidth_mmd_is_symmetric_and_zero_for_identical_samples():
    rng=np.random.default_rng(8); a=rng.normal(size=(12,4)); b=rng.normal(size=(7,4))
    assert abs(mmd_fixed_bandwidth(a,a,2))<1e-12
    assert np.isclose(mmd_fixed_bandwidth(a,b,2),mmd_fixed_bandwidth(b,a,2))


def test_constrained_convex_respects_fitted_bounds_and_budget():
    rng=np.random.default_rng(3); X=rng.normal(size=(12,4)); g=ConditionalPCAGaussian(3,2).fit(X,np.repeat('A',12))
    real=g.prep.named_steps['scale'].inverse_transform(g.prep.transform(X))
    a,info=constrained_convex(real,g,[0],20,rng)
    assert len(a)==20 and info['attempts']<=5000
    assert np.all(a>=g.low-1e-9) and np.all(a<=g.high+1e-9)


def test_saved_graph_comparison_contains_all_15_folds_and_ten_null_replicates():
    path=ROOT/'outputs/stage3/q2/graph_fold_metrics.csv'
    if not path.exists(): pytest.skip('Third-stage graph study not present')
    d=pd.read_csv(path); assert d.fold.nunique()==15
    shuffled=d[d.graph=='shuffled']; assert len(shuffled)==300
    assert (shuffled.groupby(['fold','scope']).size()==10).all()


def test_saved_search_evaluator_and_test_roles_are_disjoint():
    path=ROOT/'outputs/stage3/q3/disjoint_roles.csv'
    if not path.exists(): pytest.skip('Third-stage generation study not present')
    d=pd.read_csv(path)
    for _,a in d.groupby('repeat'):
        assert len(a)==a.beer.nunique()==250
        assert sorted(a.groupby('role').size())==[50,100,100]


def test_saved_failed_transfers_are_not_counted_as_improvements():
    path=ROOT/'outputs/stage3/q3/transfer_independent_evaluation.csv'
    if not path.exists(): pytest.skip('Third-stage transfer study not present')
    d=pd.read_csv(path); failed=d.status=='no_feasible_nontrivial_candidate_source_retained'
    assert not d.loc[failed,'evaluator_proxy_acceptance'].any()
    assert (d.loc[failed,'standardized_change']==0).all()


def test_missing_sources_are_never_reported_as_unconditional_feasible():
    path=ROOT/'outputs/stage3/q3/transfer_independent_evaluation.csv'
    if not path.exists(): pytest.skip('Third-stage transfer study not present')
    d=pd.read_csv(path)
    assert not ((~d.all_provided_source_defects_observed)&(d.status=='model_feasible_nontrivial')).any()
    assert not d.absolute_chemical_certification.any() and not d.real_tasting_validated.any()


def test_saved_ordinal_calibration_roles_are_disjoint():
    paths=list((ROOT/'outputs/stage3/q1').glob('conformal_roles_*.csv'))
    if len(paths)!=5: pytest.skip('Third-stage conformal experiments not complete')
    for path in paths:
        d=pd.read_csv(path); assert len(d)==d.beer.nunique()==250
        assert d.groupby('role').size().to_dict()=={'calibration':50,'test':50,'train':150}


def test_saved_prediction_study_is_complete_and_absolute_scores_remain_unidentified():
    path=ROOT/'outputs/stage3/q1/status.json'
    if not path.exists(): pytest.skip('Third-stage prediction study not complete')
    d=json.loads(path.read_text()); assert d['complete'] and d['outer_folds']==15
    assert not d['absolute_anchors_observed']
