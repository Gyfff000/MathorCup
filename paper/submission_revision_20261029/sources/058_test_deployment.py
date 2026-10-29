"""Exercise model packaging and prediction on unseen files without sensory inputs."""
import numpy as np
import pandas as pd
import pytest
from src.deployment import fit_final, predict, read_input


def test_final_model_predicts_from_unlabelled_reordered_chemical_input(tmp_path):
    n = 14
    chem = pd.DataFrame({'beer': np.arange(n), 'tasting_category_fine': ['a'] * n,
                         'beer_id': np.arange(n), 'x1': np.linspace(-1, 1, n), 'x2': np.arange(n) % 3})
    means = pd.DataFrame({'beer': np.arange(n), 'taste': np.arange(n) % 6})
    prob = pd.DataFrame({'beer': np.arange(n), 'target': ['taste'] * n})
    for k in range(7): prob[f'observed_panel_p{k}'] = (means.taste == k).astype(float)
    for path, frame in [('chemical.csv', chem), ('means.csv', means), ('q.csv', prob)]:
        frame.to_csv(tmp_path / path, index=False)
    config = {'seed': 11, 'extra_trees': 5, 'gbr_estimators': 5}
    model = fit_final(tmp_path / 'chemical.csv', tmp_path / 'means.csv', tmp_path / 'q.csv', config)
    unlabelled = chem[['beer', 'x2', 'x1']].iloc[:3].copy()
    unlabelled.loc[1, 'x1'] = np.nan
    unlabelled.to_csv(tmp_path / 'new.csv', index=False)
    result = predict(model, tmp_path / 'new.csv')
    assert list(result['panel_mean'].columns) == ['beer', 'taste']
    assert result['panel_mean'].taste.between(0, 6).all()
    assert result['modal_grade'].taste.between(0, 6).all()
    assert np.allclose(result['rating_distribution'].filter(regex=r'^p\d$').sum(axis=1), 1)
    assert 'taste' not in unlabelled.columns
    with pytest.raises(ValueError, match='Missing'):
        read_input(tmp_path / 'new.csv', ['x1', 'missing'])


def test_duplicate_and_infinite_input_is_rejected(tmp_path):
    (tmp_path / 'duplicate.csv').write_text('beer,x,x\n1,1,2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        read_input(tmp_path / 'duplicate.csv', ['x'])
    (tmp_path / 'infinite.csv').write_text('beer,x\n1,inf\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Infinite'):
        read_input(tmp_path / 'infinite.csv', ['x'])
