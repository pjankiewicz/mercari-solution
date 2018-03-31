import sys

from mercari.config import TEST_SIZE
from mercari.datasets_tf import prepare_vectorizer_1_tf, prepare_vectorizer_3_tf, prepare_vectorizer_2_tf
from mercari.main_helpers import main
from mercari.tf_sparse import RegressionHuber, RegressionClf, prelu


def define_models_1(n_jobs, seed):
    h0 = 192  # the same to make training take the same time
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu)
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_1_tf(n_jobs=n_jobs)


def define_models_2(n_jobs, seed):
    h0 = 192  # the same to make training take the same time
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu, binary_X=True),
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu, binary_X=True),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu)
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_2_tf(n_jobs=n_jobs)


def define_models_3(n_jobs, seed):
    h0 = 128  # reduced from 192 due to kaggle slowdown
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu, binary_X=True),
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu, binary_X=True),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu),
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_3_tf(n_jobs)


if __name__ == '__main__':
    main(
        'tf',
        sys.argv[1],
        {
            1: define_models_1(n_jobs=4, seed=1),
            2: define_models_2(n_jobs=4, seed=2),
            3: define_models_3(n_jobs=4, seed=3),
        },
    )
