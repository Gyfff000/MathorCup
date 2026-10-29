"""Meaningful property tests for the new mathematical helpers."""
import numpy as np
import pytest
from src.stage3_algorithms import (fixed_grade_boundaries, grade_with_fixed_boundaries,
    ordinal_interval_image, shrunk_blend_weights, paired_oof_bootstrap, pareto_minimize, threshold_excess)


def test_fixed_ordinal_map_preserves_contained_outcome_with_tied_boundaries():
    boundaries = fixed_grade_boundaries(np.array([[0,0],[0,1],[0,2],[1,3],[1,4],[1,5],[2,6]], float))
    lower, upper = np.array([[-1,1.5],[0,2]]), np.array([[2,4],[0,2]])
    lo, hi = ordinal_interval_image(lower, upper, boundaries)
    for weight in np.linspace(0,1,21):
        inside = grade_with_fixed_boundaries(lower+weight*(upper-lower),boundaries)
        assert np.all((lo <= inside) & (inside <= hi))


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError):
        ordinal_interval_image(np.ones((1,2)), np.zeros((1,2)), np.zeros((6,2)))


def test_blend_recovers_opposite_best_models_and_shrinks_to_global():
    y = np.zeros((20,2)); et = np.tile([0,1.],(20,1)); gbr=np.tile([1.,0],(20,1))
    w,_ = shrunk_blend_weights(y,et,gbr,0)
    np.testing.assert_allclose(w,[0,1])
    w,global_w = shrunk_blend_weights(y,et,gbr,1e8)
    np.testing.assert_allclose(w,global_w,atol=1e-7)


def test_fixed_prediction_resampling_uses_paired_errors_and_declares_scope():
    y=np.zeros((12,2)); a=np.zeros_like(y); b=np.ones_like(y)
    r=paired_oof_bootstrap(y,a,b,np.repeat(['A','B'],6),100)
    assert r['difference_A_minus_B'] == -1
    assert r['conditional_percentile_95'] == [-1,-1]
    assert 'does_not_include_model_retraining' in r['scope']


def test_pareto_retains_tradeoffs_and_rejects_nonfinite_candidate():
    keep=pareto_minimize([[0,2],[1,1],[2,0],[2,2],[np.nan,0],[1,1]])
    np.testing.assert_array_equal(keep,[True,True,True,False,False,True])


def test_unknown_threshold_and_source_values_never_become_safe_zero():
    result=threshold_excess([.5,2,2,np.nan,-1],[1,1,np.nan,1,1])
    np.testing.assert_allclose(result[:2],[0,1])
    assert np.isnan(result[2:]).all()
