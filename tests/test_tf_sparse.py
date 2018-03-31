import numpy as np
from scipy.sparse import csr_matrix

from mercari.tf_sparse import Regression


def test_random_seed():
    X = csr_matrix(np.random.random((100, 100)))
    y = np.random.random(100)
    reg = Regression(seed=0)
    reg.fit(X, y)
    pred_1 = reg.predict(X)

    reg = Regression(seed=0)
    reg.fit(X, y)
    pred_2 = reg.predict(X)

    reg = Regression(seed=1)
    reg.fit(X, y)
    pred_3 = reg.predict(X)

    assert np.allclose(pred_1, pred_2)
    assert not np.allclose(pred_1, pred_3)
