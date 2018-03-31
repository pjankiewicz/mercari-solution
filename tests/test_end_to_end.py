import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from mercari.datasets_mx import prepare_vectorizer_1, prepare_vectorizer_2, prepare_vectorizer_3
from mercari.datasets_tf import prepare_vectorizer_1_tf, prepare_vectorizer_2_tf, prepare_vectorizer_3_tf
from mercari.mercari_io import load_train
from mercari.mx_sparse import MXRegression, MXRegressionClf
from mercari.tf_sparse import RegressionHuber, RegressionClf
from mercari.utils import rmsle


@pytest.mark.parametrize('vectorizer', [
    prepare_vectorizer_1(),
    prepare_vectorizer_2(),
    prepare_vectorizer_3(),
    prepare_vectorizer_1_tf(),
    prepare_vectorizer_2_tf(),
    prepare_vectorizer_3_tf(),
])
@pytest.mark.parametrize('model', [
    MXRegression(n_epoch=3, loss='huber'),
    MXRegression(n_epoch=3, binary_X=True, loss='huber'),
    MXRegressionClf(n_epoch=3, n_hidden=(196, 64)),
    MXRegressionClf(n_epoch=3, n_hidden=(196, 64), binary_X=True),
    RegressionHuber(n_epoch=3),
    RegressionHuber(n_epoch=3, binary_X=True),
    RegressionClf(n_epoch=3, n_hidden=(196, 64)),
    RegressionClf(n_epoch=3, n_hidden=(196, 64), binary_X=True)
])
def test_end_to_end(vectorizer, model):
    _test(vectorizer, model, n_rows=None)


@pytest.mark.parametrize('model', [
    MXRegression(n_epoch=3, loss='huber'),
    MXRegressionClf(n_epoch=3, n_hidden=(196, 64)),
    RegressionHuber(n_epoch=3),
    RegressionClf(n_epoch=3, n_hidden=(196, 64)),
])
@pytest.mark.parametrize('vectorizer', [
    prepare_vectorizer_1(),
    prepare_vectorizer_1_tf(),
])
@pytest.mark.parametrize('n_rows', [
    None,
    'random',
    1,
    2,
    2**10,
    2**13 - 1,
    2**13,
    2**13 + 1,
    2**13 + 2**10,
])
def test_random_number_of_rows(vectorizer, model, n_rows):
    _test(vectorizer, model, n_rows)


def _test(vectorizer, model, n_rows):
    tr = load_train('tests/train_10k.tsv')
    tr, va = train_test_split(tr)
    te = pd.read_csv('tests/test_10k_corrupted.tsv', sep="\t")
    if n_rows is not None:
        if n_rows == 'random':
            n_rows = np.random.randint(1, te.shape[0])
            te = te.sample(n=n_rows)
    mat_tr = vectorizer.fit_transform(tr, tr.price)
    mat_te = vectorizer.transform(te.copy())
    mat_va = vectorizer.transform(va)
    model.fit(mat_tr, np.log1p(tr.price))
    assert rmsle(np.expm1(model.predict(mat_va)), va.price) < 0.85
    te_preds = np.expm1(model.predict(mat_te))
    assert te_preds.shape[0] == te.shape[0]
    assert np.all(np.isfinite(te_preds))
    assert te_preds.min() >= -1, "min price is {}".format(te_preds.min())
    assert te_preds.max() <= 3000, "max price is {}".format(te_preds.max())
