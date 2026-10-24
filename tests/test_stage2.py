"""AI-assisted validation: OpenAI Codex / GPT-6, 2026-10-24."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stage2 import assumption_grades, conformal_quantile, local_convex_samples

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage2"


def test_conformal_quantile_uses_finite_sample_rank():
    errors = np.arange(1, 51)[:, None]
    assert conformal_quantile(errors, .1)[0] == 46
    with pytest.raises(ValueError):
        conformal_quantile(errors[:2], .01)


def test_three_grade_assumptions_remain_bounded():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(50, 4))
    grades = assumption_grades(ref, np.array([[-100.] * 4, [100.] * 4]))
    assert grades.shape == (3, 2, 4)
    assert np.all((grades >= 0) & (grades <= 6))
    assert np.all(grades[:, 0] <= grades[:, 1])


def test_local_convex_samples_stay_inside_training_hull_box():
    real = np.array([[0., 1.], [1., 2.], [2., 3.]])
    generated = local_convex_samples(real, 50, np.random.default_rng(4))
    assert np.all(generated >= real.min(0)) and np.all(generated <= real.max(0))
    assert np.allclose(generated[:, 1] - generated[:, 0], 1.)
    assert not np.any(np.all(np.isclose(generated[:, None], real[None]), axis=2))


def need_stage2():
    if not (OUT / "status.json").exists():
        pytest.skip("Stage two has not been run")


def test_conformal_roles_are_disjoint():
    need_stage2()
    frame = pd.read_csv(OUT / "q1/conformal_split.csv")
    assert len(frame) == frame.beer.nunique() == 250
    assert frame.role.value_counts().to_dict() == {"train": 150, "calibration": 50, "test": 50}


def test_attribution_flags_do_not_claim_causality():
    need_stage2()
    top = pd.read_csv(OUT / "q2/top3_evidence_review.csv")
    assert len(top) == 150
    selected = top[top.evidence_status == "comparatively_supported_association"]
    assert (selected.bootstrap_top3_frequency >= .5).all()
    assert (selected.heldout_permutation_MAE_increase > 0).all()
    assert (selected.positive_fold_fraction >= 2 / 3).all()
    assert (top.causality == "not_established").all()


def test_repeated_generator_comparison_has_all_methods_and_splits():
    need_stage2()
    d = pd.read_csv(OUT / "q3/generator_comparison_by_split.csv")
    assert len(d) == 5 * 3 * 3
    assert set(d.method) == {"bootstrap", "local_convex", "PCA_Gaussian"}
    assert np.isfinite(d[["KS_mean", "MMD2", "target_style_probability"]]).all().all()


def test_extended_transfer_preserves_failure_and_each_defect():
    need_stage2()
    d = pd.read_csv(OUT / "q3/expanded_transfer_diagnostics.csv")
    good = d[d.status == "nontrivial_model_feasible_candidate"]
    assert (good.candidate_trip_probability > good.source_trip_probability + .01).all()
    assert (good.candidate_trip_probability > good.cleaning_only_probability + .01).all()
    bad = d[d.status != "nontrivial_model_feasible_candidate"]
    assert np.allclose(bad.change_l2, 0)
    assert pd.read_csv(OUT / "q3/expanded_transfer_defect_checks.csv").nonincrease.all()


def test_stage2_keeps_absolute_calibration_unidentified():
    need_stage2()
    assert json.loads((OUT / "status.json").read_text())["absolute_score_calibration"].startswith("not_identifiable")
