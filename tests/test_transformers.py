import pytest
import numpy as np
from scipy import sparse as sp

from mercari.transformers import *


@pytest.fixture()
def df():
    return pd.DataFrame({
        'name': ['this is a name'],
        'item_description': ['This is a description'],
        'item_condition_id': ['1'],
        'shipping': [0],
        'brand_name': ['Apple']
    })


def test_text_concat(df):
    assert ConcatTexts(columns=['name', 'brand_name']).fit_transform(df)['text_concat'][
               0].strip() == 'cs000 this is a name cs001 Apple'



def test_sanitize_sparse_matrix():
    train_arr = sp.csr_matrix(np.array([
        [0,0,1],
        [0,1,0],
        [1,0,0]
    ]))

    test_arr = sp.csr_matrix(np.array([
        [0,0,2],
        [np.nan, 0, 0]
    ]))

    expected_arr = np.array([
        [0,0,1],
        [0,0,0]
    ])

    sanitize = SanitizeSparseMatrix()
    sanitize.fit(train_arr, [0]*train_arr.shape[0])

    assert np.allclose(sanitize.transform(test_arr).todense(), expected_arr)