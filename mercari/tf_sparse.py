import os

from sklearn.base import RegressorMixin, BaseEstimator

os.environ['OMP_NUM_THREADS'] = '1'
from functools import partial
import logging
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
import tensorflow as tf

from mercari.utils import log_time, memory_info
from mercari.regression_clf import (
    binarize, get_mean_percentiles, get_percentiles,
)

log_time = partial(log_time, name='tf_sparse')


def identity(x):
    return x


def sparse_linear(xs, shape, name: str, actfunc=identity):
    assert len(shape) == 2
    w = tf.get_variable(name, initializer=tf.glorot_normal_initializer(),
                        shape=shape)
    bias = tf.Variable(tf.zeros(shape[1]))
    return actfunc(tf.sparse_tensor_dense_matmul(xs, w) + bias), w


def linear(xs, shape, name: str, actfunc=identity):
    assert len(shape) == 2
    w = tf.get_variable(name, initializer=tf.glorot_normal_initializer(),
                        shape=shape)
    bias = tf.Variable(tf.zeros(shape[1]))
    output = tf.matmul(xs, w) + bias
    return actfunc(output)


class SparseMatrix:
    def __init__(self):
        self.indices = tf.placeholder(tf.int64, shape=[None, 2])
        self.values = tf.placeholder(tf.float32, shape=[None])
        self.shape = tf.placeholder(tf.int64, shape=[2])
        self.tensor = tf.SparseTensor(
            indices=self.indices,
            values=self.values,
            dense_shape=self.shape,
        )

    def feed_dict(self, X, binary_X=False):
        coo = X.tocoo()
        return {
            self.indices: np.stack([coo.row, coo.col]).T,
            self.values: coo.data if not binary_X else np.ones_like(coo.data),
            self.shape: np.array(X.shape),
        }


class Regression(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden=(16, 16),
                 learning_rate=1e-2, lr_decay=0.75,
                 n_epoch=4, batch_size=2 ** 10, bs_decay=2,
                 decay_epochs=2,
                 reg_l2=1e-5, reg_l1=0.0, use_target_scaling=True, n_bins=64, seed=0,
                 actfunc=tf.nn.relu, binary_X=False):
        self.n_hidden = n_hidden
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.bs_decay = bs_decay
        self.decay_epochs = decay_epochs
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.use_target_scaling = use_target_scaling
        if self.use_target_scaling:
            self.target_scaler = StandardScaler()
        self.n_bins = n_bins
        self.seed = seed
        self.actfunc = actfunc
        self.binary_X = binary_X
        self.is_fitted = False

    def _build_model(self, n_features: int):
        self._ys = tf.placeholder(tf.float32, shape=[None])
        self._build_hidden(n_features)
        self._output = tf.squeeze(linear(self._hidden, [self.n_hidden[-1], 1], 'l_last'), axis=1)
        self._loss = tf.losses.mean_squared_error(self._output, self._ys)
        self._add_regularization()

    def _build_hidden(self, n_features: int):
        hidden, self._w_1 = sparse_linear(
            self._xs.tensor, [n_features, self.n_hidden[0]], 'l1', self.actfunc)
        if len(self.n_hidden) == 3:
            hidden = linear(hidden, [self.n_hidden[0], self.n_hidden[1]], 'l2', self.actfunc)
        self._hidden = linear(hidden, [self.n_hidden[-2], self.n_hidden[-1]], 'l_hidden', self.actfunc)

    def _add_regularization(self):
        if self.reg_l2:
            self._loss += self.reg_l2 * tf.nn.l2_loss(self._w_1)
        if self.reg_l1:
            self._loss += self.reg_l1 * tf.reduce_sum(tf.abs(self._w_1))

    def fit(self, X, y, X_valid=None, y_valid=None, use_gpu=True, verbose=True,
            to_predict=None):
        self.is_fitted = True

        if self.use_target_scaling:
            self._scaler_fit(y)
            y = self._scale_target(y)
        if y_valid is not None:
            y_valid_scaled = self._scale_target(y_valid)
        n_features = X.shape[1]
        self._graph = tf.Graph()
        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            use_per_session_threads=1,
            allow_soft_placement=True,
            device_count={'CPU': 4, 'GPU': int(use_gpu)})
        self._session = tf.Session(graph=self._graph, config=config)
        predictions = []
        with self._graph.as_default():
            tf.set_random_seed(self.seed)
            random_state = np.random.RandomState(self.seed)
            self._xs = SparseMatrix()
            self._lr = tf.placeholder(tf.float32, shape=[])
            self._build_model(n_features)
            self._train_op = tf.train.AdamOptimizer(self._lr).minimize(self._loss)
            self._session.run(tf.global_variables_initializer())
            bs = self.batch_size
            lr = self.learning_rate
            for n_epoch in range(self.n_epoch):
                if verbose and n_epoch == 0:
                    logging.info(memory_info())
                t0 = time.time()
                n = len(y)
                indices = random_state.permutation(n)
                index_batches = [indices[idx: idx + bs]
                                 for idx in range(0, n, bs)]
                batches = ((X[batch_indices, :], y[batch_indices])
                           for batch_indices in index_batches)
                train_loss = 0
                for _x, _y in batches:
                    feed_dict = self._x_feed_dict(_x)
                    feed_dict[self._ys] = _y
                    feed_dict[self._lr] = lr
                    _, loss = self._session.run(
                        [self._train_op, self._loss], feed_dict=feed_dict)
                    train_loss += loss / len(index_batches)
                print_time = True
                if X_valid is not None:
                    assert not to_predict
                    if y_valid is not None:
                        feed_dict = self._x_feed_dict(X_valid)
                        feed_dict[self._ys] = y_valid_scaled
                        valid_pred, valid_loss = self._session.run(
                            [self._output, self._loss], feed_dict)
                        valid_pred = self._invert_target(valid_pred)
                        valid_rmsle = get_rmsle(y_valid, valid_pred)
                        dt = time.time() - t0
                        print_time = False
                        logging.info(
                            f'{n_epoch} train_loss: {train_loss:.5f}, '
                            f'valid_loss: {valid_loss:.5f}, '
                            f'valid rsmle: {valid_rmsle:.5f}, '
                            f'time {dt:.1f} s')
                    else:
                        valid_pred = self.predict(X_valid)
                    predictions.append(valid_pred)
                elif to_predict:
                    predictions.append([self.predict(x) for x in to_predict])
                if print_time:
                    dt = time.time() - t0
                    logging.info(f'{n_epoch} train_loss: {train_loss:.5f}, '
                                 f'time {dt:.1f} s')
                if n_epoch < self.decay_epochs:
                    bs *= self.bs_decay
                    lr *= self.lr_decay
                if verbose:
                    logging.info(memory_info())
        return self

    def _x_feed_dict(self, X):
        return self._xs.feed_dict(X, self.binary_X)

    def predict(self, X, batch_size=2 ** 13):
        assert self.is_fitted, "Model is not fitted - cannot predict"

        ys = []
        for idx in range(0, X.shape[0], batch_size):
            ys.extend(self._session.run(
                self._output, self._x_feed_dict(X[idx: idx + batch_size])))
        return self._invert_target(ys)

    def predict_hidden(self, X, batch_size=2 ** 13):
        hidden = []
        for idx in range(0, X.shape[0], batch_size):
            hidden.append(self._session.run(
                self._hidden, self._x_feed_dict(X[idx: idx + batch_size])))
        return np.concatenate(hidden)

    def _scaler_fit(self, y):
        self.target_scaler.fit(y.reshape(-1, 1))

    def _scale_target(self, y):
        y = np.array(y)
        if self.use_target_scaling:
            return self.target_scaler.transform(y.reshape(-1, 1))[:, 0]
        return y

    def _invert_target(self, y):
        y = np.array(y)
        if self.use_target_scaling:
            return self.target_scaler.inverse_transform(y.reshape(-1, 1))[:, 0]
        return y


class RegressionHuber(Regression):
    def build_model(self, n_features: int):
        super()._build_model(n_features)
        self._loss = tf.losses.huber_loss(self._output, self._ys,
                                          weights=2.0, delta=1.0)


class RegressionClf(Regression):
    def _build_model(self, n_features: int):
        self._ys = tf.placeholder(tf.float32, shape=[None, self.n_bins])
        self._build_hidden(n_features)
        logits = linear(self._hidden, [self.n_hidden[-1], self.n_bins], 'l_last')
        self._output = tf.nn.softmax(logits)
        loss_scale = 6 / self.n_bins  # 4 for 32 bins
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self._ys)) * loss_scale
        self._add_regularization()

    def _scale_target(self, ys):
        return binarize(np.array(ys), self._get_percentiles(ys))

    def _get_percentiles(self, ys):
        if not hasattr(self, 'percentiles'):
            self.percentiles = get_percentiles(ys, self.n_bins)
        return self.percentiles

    def _invert_target(self, ys):
        mean_percentiles = get_mean_percentiles(self.percentiles)
        return (mean_percentiles * ys).sum(axis=1)


def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


def prelu(_x):
    alphas = tf.get_variable('prelu_alpha_{}'.format(_x.get_shape()[-1]), _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def swish(x, name='swish'):
    """The Swish function, see `Swish: a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941>`_.
    Parameters
    ----------
    x : a tensor input
        input(s)
    Returns
    --------
    A `Tensor` with the same type as `x`.
    """
    with tf.name_scope(name) as scope:
        x = tf.nn.sigmoid(x) * x
    return x
