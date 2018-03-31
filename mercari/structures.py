import numpy as np

from mercari.config import MIN_PRICE, MAX_PRICE


class Preds():
    def __init__(self, tr_pred, va_pred, te_pred, elapsed,
                 tr_pred_res=None, va_pred_res=None, te_pred_res=None,
                 tr_tr_size=None, va_tr_size=None, te_tr_size=None,
                 clip=True, dtype=np.float32):
        self.dtype = dtype
        if clip:
            self.tr_pred = np.clip(tr_pred, MIN_PRICE, MAX_PRICE).astype(dtype)
            self.va_pred = np.clip(va_pred, MIN_PRICE, MAX_PRICE).astype(dtype)
            self.te_pred = np.clip(te_pred, MIN_PRICE, MAX_PRICE).astype(dtype)
        else:
            self.tr_pred = tr_pred.astype(dtype)
            self.va_pred = va_pred.astype(dtype)
            self.te_pred = te_pred.astype(dtype)
        self.elapsed = elapsed
        self.tr_pred_res = tr_pred_res.astype(dtype) if tr_pred_res is not None else None
        self.va_pred_res = va_pred_res.astype(dtype) if va_pred_res is not None else None
        self.te_pred_res = te_pred_res.astype(dtype) if te_pred_res is not None else None
        self.tr_tr_size = tr_tr_size
        self.va_tr_size = va_tr_size
        self.te_tr_size = te_tr_size

    def __mul__(self, c):
        return Preds(self.tr_pred * c, self.va_pred * c, self.te_pred * c,
                     self.elapsed, clip=False)

    def __add__(self, other):
        return Preds(self.tr_pred + other.tr_pred,
                     self.va_pred + other.va_pred,
                     self.te_pred + other.te_pred,
                     self.elapsed + other.elapsed,
                     clip=False)

    def map(self, fun):
        return Preds(fun(self.tr_pred),
                     fun(self.va_pred),
                     fun(self.te_pred),
                     self.elapsed,
                     clip=False)
