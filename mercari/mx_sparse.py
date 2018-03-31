import os
os.environ['OMP_NUM_THREADS'] = '1'
from functools import partial
import time

import mxnet as mx
import numpy as np
import scipy.sparse
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler

from mercari.utils import rmsle, log_time
from mercari.regression_clf import (
    binarize, get_mean_percentiles, get_percentiles,
)

log_time = partial(log_time, name='mx_sparse')


class MXRegression(BaseEstimator, RegressorMixin):

    def __init__(self, n_hidden=(196, 64, 32), n_bins=64, loss='mse',
                 learning_rate=1e-2, lr_decay=0.5,
                 n_epoch=4, batch_size=2 ** 10, bs_decay=2,
                 decay_epochs=2, reg_l2=1e-5, use_target_scaling=True,
                 binary_X=False, seed=0):

        self.n_hidden = n_hidden
        self.n_bins = n_bins
        self.loss = loss
        self.reg_l2 = reg_l2
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.bs_decay = bs_decay
        self.decay_epochs = decay_epochs
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.use_target_scaling = use_target_scaling
        if self.use_target_scaling:
            self.target_scaler = StandardScaler()
        self.binary_X = binary_X
        self.seed = seed
        self.is_fitted = False

    def _build_model(self, n_features: int):
        assert len(self.n_hidden) == 3
        xavier = mx.initializer.Xavier()
        mxs = mx.symbol
        X = mxs.Variable('data', stype='csr')
        Y = mxs.Variable('label')

        w1 = mxs.Variable('w1', stype='row_sparse',
                          shape=(n_features, self.n_hidden[0]),
                          init=xavier, wd_mult=1)
        w2 = mxs.Variable('w2', shape=(self.n_hidden[0], self.n_hidden[1]),
                          init=xavier, wd_mult=0)
        w3 = mxs.Variable('w3', shape=(self.n_hidden[1], self.n_hidden[2]),
                          init=xavier, wd_mult=0)
        w4 = mxs.Variable('w4', shape=(self.n_hidden[2], 1),
                          init=xavier, wd_mult=0)

        zero = mx.init.Zero()
        b1 = mxs.Variable('b1', shape=(self.n_hidden[0],), init=zero, wd_mult=0)
        b2 = mxs.Variable('b2', shape=(self.n_hidden[1],), init=zero, wd_mult=0)
        b3 = mxs.Variable('b3', shape=(self.n_hidden[2],), init=zero, wd_mult=0)
        b4 = mxs.Variable('b4', shape=(1,), init=zero, wd_mult=0)

        x = mxs.sparse.dot(X, w1)
        x = mxs.broadcast_add(x, b1)
        x = mxs.relu(x)
        x = mxs.dot(x, w2)
        x = mxs.broadcast_add(x, b2)
        x = mxs.relu(x)
        x = mxs.dot(x, w3)
        x = mxs.broadcast_add(x, b3)
        x = mxs.relu(x)
        x = mxs.dot(x, w4)
        x = mxs.broadcast_add(x, b4, name='regression_output')
        output = x

        if self.loss == 'mse':
            loss = mxs.LinearRegressionOutput(data=output, label=Y, name='loss')
        elif self.loss == 'huber':
            rho = 1.0
            loss = mxs.abs(Y - mxs.reshape(output, shape=0))
            loss = mxs.where(loss > rho, loss - 0.5 * rho,
                             (0.5 / rho) * mxs.square(loss))
            loss = mxs.MakeLoss(loss, name='loss')
        else:
            raise ValueError(f'Unknown loss {self.loss}')
        return loss, output

    def _make_train_iter(self, X, y, batch_size, shuffle):
        return CSRIter(
            X, y, batch_size, shuffle=shuffle,
            binary_X=self.binary_X,
            last_batch_handle='discard',
            label_name='label')

    @log_time
    def fit(self, X, y, X_valid=None, y_valid=None, eval_train=False):
        mx.random.seed(self.seed)
        np.random.seed(self.seed)

        self.is_fitted = True
        y_train = y
        if self.use_target_scaling:
            self._scaler_fit(y)
            y = self._scale_target(y)
        n_features = X.shape[1]

        loss, self.output = self._build_model(n_features=n_features)
        make_train_iter = lambda bs: self._make_train_iter(
            X, y, batch_size=bs, shuffle=True)
        train_iter = make_train_iter(self.batch_size)
        mod = mx.mod.Module(
            symbol=loss, data_names=['data'], label_names=['label'])
        mod.bind(
            data_shapes=train_iter.provide_data,
            label_shapes=train_iter.provide_label,
            for_training=True,
        )

        initializer = mx.initializer.Xavier()
        mod.init_params(initializer=initializer)

        optimizer = mx.optimizer.Adam(
            learning_rate=self.learning_rate, wd=self.reg_l2)
        mod.init_optimizer(optimizer=optimizer)
        metric = mx.metric.create('MSE')  # see comment below

        lr = self.learning_rate
        bs = self.batch_size
        for n_epoch in range(self.n_epoch):
            optimizer.lr = lr
            train_iter.reset()
            t0 = time.time()
            for batch in train_iter:
                mod.forward_backward(batch)
                mod.update()
                # this is added only to sync gradient updates
                mod.update_metric(metric, batch.label)
            mod._sync_params_from_devices()

            if eval_train:
                train_rmsle = rmsle(np.expm1(self.predict(X)),
                                    np.expm1(y_train))
            else:
                train_rmsle = 0
            if y_valid is not None:
                valid_rmsle = rmsle(np.expm1(self.predict(X_valid)),
                                    np.expm1(y_valid))
                dt = time.time() - t0
                print(f'Epoch {n_epoch}, train RMSLE {train_rmsle:.4f}, '
                      f'valid RMSLE {valid_rmsle:.4f}, time {dt:.1f} s')
            else:
                dt = time.time() - t0
                print(f'Epoch {n_epoch}, train RMSLE {train_rmsle:.4f} '
                      f'time {dt:.1f} s')
            if n_epoch < self.decay_epochs:
                bs *= self.bs_decay
                lr *= self.lr_decay
            train_iter = make_train_iter(bs)
            mod.bind(
                data_shapes=train_iter.provide_data,
                label_shapes=train_iter.provide_label,
                for_training=True,
                force_rebind=True,
            )
        self.mod_params = mod.get_params()

        return self

    @log_time
    def predict(self, X):
        assert self.is_fitted
        ys = []
        n = X.shape[0]
        batch_size = min(n, 2**13)
        mod = mx.mod.Module(symbol=self.output, label_names=None)
        for _ in range(2):
            eval_iter = self._make_train_iter(
                X[len(ys):, :], y=None, batch_size=batch_size, shuffle=False)
            mod.bind(
                data_shapes=eval_iter.provide_data,
                label_shapes=None,
                for_training=False,
                force_rebind=True,
            )
            mod.set_params(*self.mod_params)
            ys.extend(mod.predict(eval_iter).asnumpy())
            batch_size = n % batch_size
            if batch_size == 0:
                break
        assert len(ys) == n
        return self._invert_target(ys)

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


class MXRegressionClf(MXRegression):

    def _build_model(self, n_features: int):
        assert len(self.n_hidden) == 2
        xavier = mx.initializer.Xavier()
        mxs = mx.symbol
        X = mxs.Variable('data', stype='csr')
        Y = mxs.Variable('label')

        w1 = mxs.Variable('w1', stype='row_sparse',
                          shape=(n_features, self.n_hidden[0]),
                          init=xavier, wd_mult=1)
        w2 = mxs.Variable('w2', shape=(self.n_hidden[0], self.n_hidden[1]),
                          init=xavier, wd_mult=0)
        w3 = mxs.Variable('w3', shape=(self.n_hidden[1], self.n_bins),
                          init=xavier, wd_mult=0)

        zero = mx.init.Zero()
        b1 = mxs.Variable('b1', shape=(self.n_hidden[0],), init=zero, wd_mult=0)
        b2 = mxs.Variable('b2', shape=(self.n_hidden[1],), init=zero, wd_mult=0)
        b3 = mxs.Variable('b3', shape=(self.n_bins,), init=zero, wd_mult=0)

        x = mxs.sparse.dot(X, w1)
        x = mxs.broadcast_add(x, b1)
        x = mxs.relu(x)
        x = mxs.dot(x, w2)
        x = mxs.broadcast_add(x, b2)
        x = mxs.relu(x)
        x = mxs.dot(x, w3)
        x = mxs.broadcast_add(x, b3)
        output = mxs.softmax(x)

        loss_scale = 6 / self.n_bins  # 4 for 32 bins
        loss = mxs.SoftmaxOutput(data=x, label=Y, name='loss',
                                 grad_scale=loss_scale)
        return loss, output

    def _scale_target(self, ys):
        return binarize(np.array(ys), self._get_percentiles(ys))

    def _get_percentiles(self, ys):
        if not hasattr(self, 'percentiles'):
            self.percentiles = get_percentiles(ys, self.n_bins)
        return self.percentiles

    def _invert_target(self, ys):
        mean_percentiles = get_mean_percentiles(self.percentiles)
        return (mean_percentiles * ys).sum(axis=1)


class CSRIter(mx.io.DataIter):
    def __init__(self, X, y=None, batch_size=1, shuffle=False,
                 data_name='data', label_name='label',
                 last_batch_handle='discard', binary_X=False):
        assert isinstance(X, scipy.sparse.csr_matrix)
        super().__init__(batch_size=batch_size)
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.data_name = data_name
        self.label_name = label_name
        self.binary_X = binary_X
        assert last_batch_handle == 'discard'

    def reset(self):
        n = self.X.shape[0]
        bs = self.batch_size
        if bs <= n:
            if self.shuffle:
                indices = np.random.permutation(n)
                self._index_batches = [indices[idx: idx + bs]
                                       for idx in range(0, n - bs + 1, bs)]
            else:
                self._index_batches = [slice(idx, idx + bs)
                                       for idx in range(0, n - bs + 1, bs)]
        else:
            self._index_batches = []
        self._idx = None

    def iter_next(self):
        if not self._index_batches:
            return False
        elif self._idx is None:
            self._idx = 0
            return True
        elif self._idx < len(self._index_batches) - 1:
            self._idx += 1
            return True
        return False

    def getdata(self):
        X = self.X[self._index_batches[self._idx], :]
        if self.binary_X:
            X = X.astype(np.bool)
        X = X.astype(np.float32)
        return [mx.nd.sparse.array(X)]

    def getlabel(self):
        if self.y is not None:
            y = self.y[self._index_batches[self._idx]]
            y = y.astype(np.float32)
            return [mx.nd.array(y)]
        else:
            return []

    def getpad(self):
        return 0

    def getindex(self):
        return self._index_batches[self._idx]

    @property
    def provide_data(self):
        return [mx.io.DataDesc(
            self.data_name, (self.batch_size,) + self.X.shape[1:],
            np.float32)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc(
            self.label_name, (self.batch_size,) + self.y.shape[1:],
            np.float32)]
