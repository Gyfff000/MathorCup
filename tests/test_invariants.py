"""AI-assisted verification: OpenAI Codex / GPT-6, 2026-10-24."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold

from src.common import TrainQuantileClipper, ecdf_grades, sha256
from src.data import CHEMICAL_DEFECTS, NON_MOLECULES, align_by_beer, find_data, load_data
from src.q2 import signed_laplacian
from src.q3 import ConditionalPCAGaussian, chemical_nonincrease


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/stage1"


def test_join_uses_unique_beer_and_recovers_permuted_rows():
    x = pd.DataFrame({"beer": [1, 2], "beer_id": [100, 101], "tasting_category_fine": ["a", "b"]})
    y = pd.DataFrame({"beer": [2, 1], "beer_id": [100, 100], "tasting_category_fine": ["b", "a"], "score": [9, 4]})
    assert align_by_beer(x, y).score.tolist() == [4, 9]


def test_duplicate_key_is_rejected_before_join():
    a = pd.DataFrame({"beer": [1, 1], "tasting_category_fine": ["a", "a"]})
    with pytest.raises(ValueError, match="unique"):
        align_by_beer(a, a)


def test_preprocessing_bounds_do_not_learn_from_validation_extreme():
    train = np.array([[0.], [1.], [2.], [3.]])
    clip = TrainQuantileClipper(lower=0, upper=1).fit(train)
    assert clip.transform([[1e12]])[0, 0] == 3
    np.testing.assert_array_equal(clip.upper_, [3.])


def test_signed_laplacian_is_psd_with_negative_edges():
    L = signed_laplacian([[1, -.8, .7], [-.8, 1, .1], [.7, .1, 1]], .6)
    assert np.linalg.eigvalsh(L).min() > -1e-10


def test_ecdf_grades_are_bounded_and_monotone():
    ref = np.arange(100.)[:, None]
    g = ecdf_grades(ref, np.array([[-1.], [50.], [1000.]]))[:, 0]
    assert g[0] == 0 and g[-1] == 6 and np.all(np.diff(g) >= 0)


def test_each_defect_is_checked_separately():
    # A falling total must not hide a rise in one defect.
    source, candidate = np.array([2., 2.]), np.array([0., 3.])
    assert candidate.sum() < source.sum()
    assert not chemical_nonincrease(source, candidate, [0, 1])


def test_conditional_generator_is_finite_bounded_and_reproducible():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 7))
    x[0, 0] = np.nan
    gen = ConditionalPCAGaussian(3).fit(x, np.array(["a"] * 15 + ["b"] * 15))
    a = gen.sample("a", 10, np.random.default_rng(1))
    b = gen.sample("a", 10, np.random.default_rng(1))
    np.testing.assert_allclose(a, b)
    assert np.isfinite(a).all()
    assert np.all(a >= gen.low - 1e-10) and np.all(a <= gen.high + 1e-10)


@pytest.fixture(scope="module")
def real_data():
    try:
        root = find_data()
    except FileNotFoundError:
        pytest.skip("Original competition data not available")
    return load_data(root)


def need_results():
    if not (OUTPUT / "q3/status.json").exists():
        pytest.skip("Run all stages before artifact validation")


def test_original_sources_unchanged(real_data):
    need_results()
    manifest = pd.read_csv(OUTPUT / "audit/source_manifest.csv")
    for row in manifest.itertuples():
        assert sha256(real_data.root / row.file) == row.sha256


def test_oof_mean_matches_actual_training_only_mean(real_data):
    need_results()
    cfg = json.loads((ROOT / "config.json").read_text())
    Y = real_data.Y.to_numpy()
    expected = np.zeros_like(Y)
    counts = np.zeros(len(Y))
    splitter = RepeatedStratifiedKFold(n_splits=cfg["outer_folds"], n_repeats=cfg["outer_repeats"], random_state=cfg["seed"])
    for train, test in splitter.split(real_data.X, real_data.meta.tasting_category_fine):
        expected[test] += Y[train].mean(axis=0)
        counts[test] += 1
    saved = pd.read_csv(OUTPUT / "q1/oof_mean_continuous.csv")[real_data.Y.columns].to_numpy()
    np.testing.assert_allclose(saved, expected / counts[:, None], atol=1e-7)


def test_each_sample_held_out_once_per_repeat(real_data):
    need_results()
    s = pd.read_csv(OUTPUT / "q1/split_assignments.csv")
    assert not s.duplicated(["repeat", "beer"]).any()
    for _, part in s.groupby("repeat"):
        assert set(part.beer) == set(real_data.meta.beer)


def test_top3_has_unique_eligible_factors_per_target(real_data):
    need_results()
    top = pd.read_csv(OUTPUT / "q2/top3_by_sensory_dimension.csv")
    assert len(top) == 3 * real_data.Y.shape[1]
    assert set(top.target) == set(real_data.Y.columns)
    assert not top.duplicated(["target", "field"]).any()
    assert not set(top.field) & NON_MOLECULES
    assert top.bootstrap_top3_frequency.between(0, 1).all()


def test_transfer_failure_is_not_hidden_as_success():
    need_results()
    t = pd.read_csv(OUTPUT / "q3/transfer_diagnostics.csv")
    failed = t[t.status == "no_nontrivial_candidate_found_source_retained"]
    assert np.allclose(failed.supplied_scale_change_l2, 0)
    success = t[t.status == "nontrivial_model_feasible_candidate"]
    assert (success.candidate_target_probability > success.source_target_probability + .01).all()
    assert (success.candidate_target_probability > success.cleaning_only_target_probability + .01).all()
    assert t.chemical_nonincrease.all() and t.predicted_stale_hops_nonincrease.all()
    checks = pd.read_csv(OUTPUT / "q3/defect_nonincrease_checks.csv")
    assert checks.nonincrease.all()
    assert set(checks.defect) == set(CHEMICAL_DEFECTS) | {"stale_hops"}


def test_calibration_remains_explicitly_unresolved():
    need_results()
    status = json.loads((OUTPUT / "q1/status.json").read_text())
    assert status["absolute_0_6_prediction"].startswith("UNRESOLVED")
    grades = pd.read_csv(OUTPUT / "q1/ASSUMPTION_percentile_grades_0_6.csv")
    assert grades.status.str.startswith("ASSUMPTION_ONLY").all()


def test_generator_holdout_does_not_overlap_training(real_data):
    need_results()
    split = pd.read_csv(OUTPUT / "q3/generator_split.csv")
    train = set(split[split.role == "train"].beer)
    heldout = set(split[split.role == "holdout"].beer)
    assert train.isdisjoint(heldout)
    assert train | heldout == set(real_data.meta.beer)


def test_generation_diagnostics_do_not_hide_missing_reference_values():
    need_results()
    diagnostics = pd.read_csv(OUTPUT / "q3/generation_diagnostics.csv")
    assert np.isfinite(diagnostics.mean_marginal_KS_holdout).all()
    assert diagnostics.mean_marginal_KS_holdout.between(0, 1).all()
    assert (diagnostics.n_KS_fields_observed > 0).all()
