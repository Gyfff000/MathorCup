"""Check identity, repeated-rater weighting and physical constraint boundaries."""
import numpy as np
import pandas as pd
from src.stage4_audit import aggregate_panel, infer_identity, transform_candidates, concentration_fields, panel_rating_distribution
from src.stage4_models import calibrated_from_training, mix_concentrations, exact_nonincrease
from src.stage4_trial import analyze_paired_trial, planned_noninferiority_power
import pytest


def test_repeat_ratings_do_not_overweight_one_panelist():
    raw = pd.DataFrame({'beer': [1, 1, 1], 'taster': [1, 1, 2], 'rating': [0, 2, 6]})
    balanced, records, _ = aggregate_panel(raw, ['rating'])
    assert balanced.loc[1, 'rating'] == 3.5
    assert records.loc[1, 'rating'] == 8 / 3
    distribution = panel_rating_distribution(raw, ['rating'], [1])
    assert np.allclose(distribution[0, 0, [0, 2, 6]], [.25, .25, .5])
    assert np.allclose(distribution.sum(-1), 1)


def test_identity_uses_attributes_despite_shuffled_published_rows():
    z = pd.DataFrame({'a': [-2., 0., 3.], 'b': [3., -1., 0.]})
    raw_id = pd.Series([1, 2, 3])
    published = pd.DataFrame({'a': [3., -2., 0.], 'b': [0., 3., -1.]}, index=[40, 90, 7])
    _, mapping = infer_identity(z, raw_id, published, ['a', 'b'])
    assert mapping.set_index('raw_beer').beer.to_dict() == {1: 90, 2: 7, 3: 40}


def test_negative_blank_signal_is_not_logged_as_negative_concentration():
    values, half = transform_candidates(np.array([-2., 0., 2.5, 10., np.nan]))
    assert half == 1.25
    assert np.allclose(values['log10_nonpositive_half_min'][:2], np.log10(1.25))
    assert np.isnan(values['log10_nonpositive_half_min'][-1])


def test_unit_and_calibration_registry_excludes_relative_area_and_pressure():
    calibrated = pd.DataFrame(columns=['beer_id', 'tasting_category_fine', 'a', 'b', 'c'])
    registry = pd.DataFrame({'field_name_R': ['a', 'b', 'c'], 'calibrated': ['yes', np.nan, 'yes'], 'unit': ['mg/L', 'A.U.', 'PSI']})
    assert concentration_fields(calibrated, registry, ['mg/L']) == ['a']


def test_isotonic_calibration_respects_only_supplied_training_anchor_range():
    result = calibrated_from_training(np.array([[0.], [1.], [2.]]), np.array([[1.], [2.], [3.]]), np.array([[-10.], [10.]]))
    assert np.array_equal(result[:, 0], [1., 3.])


def test_mixture_is_linear_in_concentration_and_missing_source_cannot_certify():
    source, anchor = np.array([1., 10.]), np.array([3., 2.])
    values = mix_concentrations(source, anchor, [.25, .5])
    assert np.allclose(values, [[1.5, 8.], [2., 6.]])
    assert not exact_nonincrease(values, source, [0, 1]).any()
    assert exact_nonincrease(values, source, [1]).all()
    assert not exact_nonincrease(values, np.array([1., np.nan]), [1]).any()


def test_trial_repeated_batches_do_not_inflate_independent_participant_count():
    rows = []
    for person in range(4):
        for batch in range(3):
            for condition in ['control', 'candidate']:
                values = dict.fromkeys(['dms', 'diacetyl', 'X4vg', 't2n', 'stale_hops'], 1.)
                rows.append({'panelist': person, 'batch': batch, 'condition': condition,
                             'style_Tripel_fit': 1. if condition == 'control' else 3., **values})
    result = analyze_paired_trial(pd.DataFrame(rows))
    assert (result.independent_panelists == 4).all() and result.criterion_passed.all()


def test_no_real_trial_claim_from_an_empty_or_unpaired_template():
    with pytest.raises(ValueError):
        analyze_paired_trial(pd.DataFrame())
    assert planned_noninferiority_power(30, .8) > planned_noninferiority_power(10, .8)
