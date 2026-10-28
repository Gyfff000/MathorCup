import numpy as np
from scripts.joint_review_experiments import risk_mix


def test_risk_mix_finds_mixture_within_all_four_bounds():
    source=np.array([10., .08, .1, .00005])
    anchors=np.array([[20., .1, .3, .00015], [12., .06, .08, .00004]])
    threshold=np.array([19., .09, .2, .0001])
    w=risk_mix(source, anchors, threshold)
    candidate=source+(anchors-source).T@w
    assert 0 < w.sum() <= .75+1e-9
    assert np.all(candidate <= np.maximum(source,threshold)+1e-10)


def test_risk_mix_returns_zero_when_each_anchor_increases_above_threshold():
    source=np.array([20., .1, .3, .0002])
    anchors=np.array([[21., .11, .4, .0003], [22., .12, .5, .0004]])
    w=risk_mix(source, anchors, np.array([19., .09, .2, .0001]))
    assert w.sum() < 1e-8
