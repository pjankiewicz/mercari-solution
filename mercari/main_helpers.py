from functools import partial
import pickle
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import traceback

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import Lasso

from mercari.config import \
    DUMP_DATASET, USE_CACHED_DATASET, DEBUG_N, HANDLE_TEST, logger, MIN_PRICE_PRED, MAX_PRICE_PRED
from mercari.mercari_io import load_train_validation, load_test_iter
from mercari.utils import Timer, rmsle


def fit_one(est, X, y):
    print("fitting y min={} max={}".format(y.min(), y.max()))
    return est.fit(X, y)


def predict_one(est, X):
    yhat = est.predict(X)
    print("predicting y min={} max={}".format(yhat.min(), yhat.max()))
    return yhat


def predict_models(X, fitted_models, vectorizer=None, parallel='thread'):
    if vectorizer:
        # TODO: parallelize this
        with Timer('Transforming data'):
            X = vectorizer.transform(X)
    predict_one_ = partial(predict_one, X=X)
    preds = map_parallel(predict_one_, fitted_models, parallel)
    return np.expm1(np.vstack(preds).T)


def fit_models(X_tr, y_tr, models, parallel='thread'):
    y_tr = np.log1p(y_tr)
    fit_one_ = partial(fit_one, X=X_tr, y=y_tr)
    return map_parallel(fit_one_, models, parallel)


def map_parallel(fn, lst, parallel, max_processes=4):
    if parallel == 'thread':
        with ThreadPool(processes=max_processes) as pool:
            return pool.map(fn, lst)
    elif parallel == 'mp':
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=max_processes) as pool:
            return pool.map(fn, lst)
    elif parallel is None:
        return list(map(fn, lst))
    else:
        raise ValueError(f'unexpected parallel value: {parallel}')


def predict_models_test_batches(models, vectorizer, parallel='thread'):
    chunk_preds = []
    test_idx = []
    for df in load_test_iter():
        test_idx.append(df.test_id.values)
        print("Predicting batch {} {}".format(df.test_id.min(), df.test_id.max()))
        chunk_preds.append(predict_models(df, models, vectorizer=vectorizer, parallel=parallel))
    predictions = np.vstack(chunk_preds)
    test_idx = np.concatenate(test_idx)
    return test_idx, predictions


def make_submission(te_idx, preds, save_as):
    submission = pd.DataFrame({
        "test_id": te_idx,
        "price": preds
    }, columns=['test_id', 'price'])
    submission.to_csv(save_as, index=False)


def fit_transform_vectorizer(vectorizer):
    df_tr, df_va = load_train_validation()
    y_tr = df_tr.price.values
    y_va = df_va.price.values
    X_tr = vectorizer.fit_transform(df_tr, y_tr)
    X_va = vectorizer.transform(df_va)
    return X_tr, y_tr, X_va, y_va, vectorizer


def fit_validate(models, vectorizer, name=None,
                 fit_parallel='thread', predict_parallel='thread'):
    cached_path = 'data_{}.pkl'.format(name)
    if USE_CACHED_DATASET:
        assert name is not None
        with open(cached_path, 'rb') as f:
            X_tr, y_tr, X_va, y_va, fitted_vectorizer = pickle.load(f)
        if DEBUG_N:
            X_tr, y_tr = X_tr[:DEBUG_N], y_tr[:DEBUG_N]
    else:
        X_tr, y_tr, X_va, y_va, fitted_vectorizer = fit_transform_vectorizer(vectorizer)
    if DUMP_DATASET:
        assert name is not None
        with open(cached_path, 'wb') as f:
            pickle.dump((X_tr, y_tr, X_va, y_va, fitted_vectorizer), f)
    fitted_models = fit_models(X_tr, y_tr, models, parallel=fit_parallel)
    y_va_preds = predict_models(X_va, fitted_models, parallel=predict_parallel)
    return fitted_vectorizer, fitted_models, y_va, y_va_preds


def merge_predictions(X_tr, y_tr, X_te=None, est=None, verbose=True):
    if est is None:
        est = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                    positive=True, random_state=9999, selection='random')
    est.fit(np.log1p(X_tr), np.log1p(y_tr))
    if hasattr(est, 'intercept_') and verbose:
        logger.info('merge_predictions = \n{:+.4f}\n{}'.format(
            est.intercept_,
            '\n'.join('{:+.4f} * {}'.format(coef, i) for i, coef in
                      zip(range(X_tr.shape[0]), est.coef_))))
    return (np.expm1(est.predict(np.log1p(X_tr))),
            np.expm1(est.predict(np.log1p(X_te))) if X_te is not None else None)


def main(name, action, arg_map, fit_parallel='thread', predict_parallel='thread'):
    prefix = lambda r: '{}_{}s'.format(name, r)

    if action in ("1", "2", "3"):
        model_round = int(action)
        models, vectorizer = arg_map[model_round]
        vectorizer, fitted_models, y_va, y_va_preds = fit_validate(
            models, vectorizer, name=model_round,
            fit_parallel=fit_parallel, predict_parallel=predict_parallel)
        joblib.dump(y_va_preds, "{}_va_preds.pkl".format(prefix(model_round)), compress=3)
        if HANDLE_TEST:
            test_idx, y_te_preds = predict_models_test_batches(
                fitted_models, vectorizer, parallel=predict_parallel)
            joblib.dump(y_te_preds, "{}_te_preds.pkl".format(prefix(model_round)), compress=3)
            joblib.dump(test_idx, "test_idx.pkl", compress=3)
        joblib.dump(y_va, "y_va.pkl", compress=3)
        for i in range(y_va_preds.shape[1]):
            print("Model {} rmsle {:.4f}".format(i, rmsle(y_va_preds[:, i], y_va)))
        print("Model mean rmsle {:.4f}".format(rmsle(y_va_preds.mean(axis=1), y_va)))

    elif action == "merge":
        va_preds = []
        te_preds = []
        for model_round in ("1", "2", "3"):
            try:
                va_preds.append(joblib.load("{}_va_preds.pkl".format(prefix(model_round))))
                if HANDLE_TEST:
                    te_preds.append(joblib.load("{}_te_preds.pkl".format(prefix(model_round))))
            except Exception as e:
                print(f'Warning: error loading round {model_round}: {e}')
                traceback.print_exc()
        va_preds = np.hstack(va_preds).clip(MIN_PRICE_PRED, MAX_PRICE_PRED)
        if HANDLE_TEST:
            te_preds = np.hstack(te_preds).clip(MIN_PRICE_PRED, MAX_PRICE_PRED)
        else:
            te_preds = None
        y_va = joblib.load("y_va.pkl")
        va_preds_merged, te_preds_merged = merge_predictions(X_tr=va_preds, y_tr=y_va, X_te=te_preds)
        print("Stacking rmsle", rmsle(y_va, va_preds_merged))
        if HANDLE_TEST:
            test_idx = joblib.load("test_idx.pkl")
            make_submission(test_idx, te_preds_merged, 'submission_merged.csv')

    elif action == "merge_describe":
        va_preds = []
        te_preds = []
        for model_round in ("1", "2", "3"):
            va_preds.append(joblib.load("{}_va_preds.pkl".format(prefix(model_round))))
            te_preds.append(joblib.load("{}_te_preds.pkl".format(prefix(model_round))))
        va_preds = np.hstack(va_preds)
        te_preds = np.hstack(te_preds)
        _, df_va = load_train_validation()
        y_va = joblib.load("y_va.pkl")
        va_preds_merged, te_preds_merged = merge_predictions(X_tr=va_preds, y_tr=y_va, X_te=te_preds)
        print("Stacking rmsle", rmsle(y_va, va_preds_merged))
        df_va['preds'] = va_preds_merged
        df_va['err'] = (np.log1p(df_va['preds']) - np.log1p(df_va['price'])) ** 2
        df_va.sort_values('err', ascending=False).to_csv('validation_preds.csv', index=False)