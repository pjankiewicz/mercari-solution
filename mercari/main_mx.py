import sys

from mercari.config import TEST_SIZE
from mercari.datasets_mx import (
    prepare_vectorizer_1, prepare_vectorizer_3, prepare_vectorizer_2,
)
from mercari.main_helpers import main
from mercari.mx_sparse import MXRegression, MXRegressionClf


def define_models_1(n_jobs, seed):
    h0 = 256  # reduced from 384 due to kaggle slowdown
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        MXRegression(
            n_hidden=(h0, 128, 64), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=1e-3, loss='huber'),
        MXRegression(
            n_hidden=(h0, 128, 64), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=1e-3, loss='huber'),
        MXRegressionClf(
            n_hidden=(h0, 128), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=2e-4),
        MXRegressionClf(
            n_hidden=(h0, 128), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=2e-4),
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_1(n_jobs=n_jobs)


def define_models_2(n_jobs, seed):
    h0 = 256
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        MXRegression(
            n_hidden=(h0, 128, 64), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=1e-3, loss='huber', binary_X=True),
        MXRegression(
            n_hidden=(h0, 128, 64), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=1e-3, loss='huber'),
        MXRegressionClf(
            n_hidden=(h0, 128), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=2e-4, binary_X=True),
        MXRegressionClf(
            n_hidden=(h0, 128), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=2e-4),
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_2(n_jobs=n_jobs)


def define_models_3(n_jobs, seed):
    h0 = 256
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        MXRegression(
            n_hidden=(h0, 128, 64), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=1e-3, loss='huber', binary_X=True),
        MXRegression(
            n_hidden=(h0, 128, 64), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=1e-3, loss='huber'),
        MXRegressionClf(
            n_hidden=(h0, 128), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=2e-4, binary_X=True),
        MXRegressionClf(
            n_hidden=(h0, 128), n_epoch=n_epoch, batch_size=2 ** 11,
            learning_rate=0.4e-2, reg_l2=2e-4),
    ]
    # 4 more same models in the best submission
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_3(n_jobs)


if __name__ == '__main__':
    main(
        'mx',
        sys.argv[1],
        {
            1: define_models_1(n_jobs=4, seed=1),
            2: define_models_2(n_jobs=4, seed=2),
            3: define_models_3(n_jobs=4, seed=3),
        },
        fit_parallel='mp',
        predict_parallel=None,
    )
