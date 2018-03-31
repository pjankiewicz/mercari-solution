import array
import re
from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _make_int_array, CountVectorizer
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm

from mercari.config import UNK, ITEM_DESCRIPTION_MAX_LENGTH, NAME_MAX_LENGTH, BRAND_NAME_MAX_LENGTH
from mercari.text_utils import extract_year, FastTokenizer, has_digit, clean_text
from mercari.utils import try_float, logger


class SparseMatrixOptimize(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def fit(self, X, *arg):
        return self

    def transform(self, X):
        return sp.csr_matrix(X, dtype=self.dtype)


class FeaturesEngItemDescription(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def transform(self, X):
        out = pd.DataFrame()
        out['lt'] = np.log1p(X['desc_clean'].str.len()) / 100.0
        out['yd'] = X['desc_clean'].map(str).map(extract_year) / 2000.0
        out['nw'] = np.log1p(X['desc_clean'].str.split().map(len))
        return out.values


class FeaturesEngName(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def transform(self, X):
        out = pd.DataFrame()
        out['lt'] = np.log1p(X['name_clean'].str.len()) / 100.0
        out['yd'] = X['name_clean'].map(str).map(extract_year) / 2000.0
        out['nw'] = np.log1p(X['name_clean'].str.split().map(len))
        return out.values


class FeaturesPatterns(BaseEstimator, TransformerMixin):
    patterns = [
        "([0-9.]+)[ ]?\$",
        "\$[ ]?([0-9.]+)",
        "paid ([0-9.]+)",
        "bought for (\d+)"
        "(10|14|18|24) gold",
        "of (\d+) ",
        " (\d+) ship",
        "is for all (\d+)",
        "is for (\d+)",
        "firm for (\d+)",
        "bundl \w+ (\d+) ",
        "(\d+) in 1",
        "^(\d+)",
        "\d+ for (\d+)",
        " x(\d+)",
        "\b(\d+)x\b",
        "(\d+)% left",
        "(\d+)[ ]?lipstick",
    ]

    def __init__(self, column):
        self.column = column

    def fit(self, X, *arg):
        return self

    def transform(self, X):
        cols = []
        self.features_names = []
        X_ = X[self.column].map(lambda x: "" if has_digit(x) else x)
        for pattern_name in tqdm(self.patterns):
            new_col = 'regex_pattern_{}_{}'.format(pattern_name, self.column)
            # TODO: parallelize this
            raw_val = X_.str.extract(pattern_name, expand=False).fillna(0)
            if isinstance(raw_val, pd.DataFrame):
                raw_val = raw_val.iloc[:, 0]
            X[new_col] = raw_val.map(try_float)
            X[X[new_col] > 2000] = 0
            X[new_col] = np.log1p(X[new_col])
            cols.append(new_col)
            self.features_names.append(new_col)
        return csr_matrix(X.ix[:, cols].values)

    def get_feature_names(self):
        return self.features_names


class PandasToRecords(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def transform(self, X):
        return X.to_dict(orient='records')


class SparsityFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_nnz=None):
        self.min_nnz = min_nnz

    def fit(self, X, y=None):
        self.sparsity = X.getnnz(0)
        return self

    def transform(self, X):
        return X[:, self.sparsity >= self.min_nnz]


class ReportShape(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info('=' * 30)
        logger.info("Matrix shape {} min {} max {}".format(X.shape, X.min(), X.max()))
        logger.info('=' * 30)
        return X


class FillEmpty(BaseEstimator, TransformerMixin):

    def fit(self, X, *args):
        return self

    def transform(self, X):
        X['name'].fillna(UNK, inplace=True)
        X['item_condition_id'] = X['item_condition_id'].fillna(UNK)
        X['category_name'].fillna(UNK, inplace=True)
        X['brand_name'].fillna(UNK, inplace=True)
        X['shipping'].fillna(0, inplace=True)
        X['item_description'].fillna(UNK, inplace=True)
        return X


class ConcatTexts(BaseEstimator, TransformerMixin):

    def __init__(self, columns, use_separators=True, output_col='text_concat'):
        self.use_separators = use_separators
        self.columns = columns
        self.output_col = output_col

    def fit(self, X, *args):
        return self

    def transform(self, X):
        X[self.output_col] = ''
        if self.use_separators:
            for i, col in enumerate(self.columns):
                X[self.output_col] += ' cs00{} '.format(i)
                X[self.output_col] += X[col]
        else:
            for i, col in enumerate(self.columns):
                X[self.output_col] += X[col]
        return X


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

        if isinstance(self.columns, str):
            self.columns = [self.columns]

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)

    def transform(self, x):
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')

        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column
        # is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1 and self.return_vector:
            return list(x[selected_cols[0]])
        else:
            return x[selected_cols]


class SGDFeatureSelectionV2(BaseEstimator, TransformerMixin):
    def __init__(self, percentile_cutoff=None):
        self.percentile_cutoff = percentile_cutoff

    def fit(self, X, y, *args):
        sgd = SGDRegressor(penalty='l1', loss='squared_loss', alpha=3.0e-11, power_t=-0.12, eta0=0.019, random_state=0,
                           average=True)
        sgd.fit(X, np.log1p(y))
        coef_cutoff = np.percentile(np.abs(sgd.coef_), self.percentile_cutoff)
        self.features_to_keep = np.where(np.abs(sgd.coef_) >= coef_cutoff)[0]
        return self

    def transform(self, X):
        return X[:, self.features_to_keep]


class FalseBrands(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def false_brand_detector(self, prefix):
        def helper(row):
            return prefix + ' ' + str(row.brand_name).lower() in str(row.item_description).lower()

        return helper

    def transform(self, X):
        return pd.DataFrame({
            'for_brand': X.apply(self.false_brand_detector('for'), axis=1),
            'like_brand': X.apply(self.false_brand_detector('like'), axis=1),
            'fits_brand': X.apply(self.false_brand_detector('fits'), axis=1)
        })


def trim_description(text):
    if text and isinstance(text, str):
        return text[:ITEM_DESCRIPTION_MAX_LENGTH]
    else:
        return text


def trim_name(text):
    if text and isinstance(text, str):
        return text[:NAME_MAX_LENGTH]
    else:
        return text


def trim_brand_name(text):
    if text and isinstance(text, str):
        return text[:BRAND_NAME_MAX_LENGTH]
    else:
        return text


class PreprocessDataKL(BaseEstimator, TransformerMixin):

    def __init__(self, num_brands, repl_patterns):
        self.num_brands = num_brands
        self.repl_patterns = repl_patterns

    def fit(self, X, y):
        self.pop_brands = X['brand_name'].value_counts().index[:self.num_brands]
        return self

    def transform(self, X):
        # fill missing values
        X['category_name'] = X['category_name'].fillna('unknown').map(str)
        X['brand_name'] = X['brand_name'].fillna('unknown').map(str)
        X['item_description'] = X['item_description'].fillna('').map(str)
        X['name'] = X['name'].fillna('').map(str)

        # trim
        X['item_description'] = X['item_description'].map(trim_description)
        X['name'] = X['name'].map(trim_name)
        X['brand_name'] = X['brand_name'].map(trim_brand_name)

        X.loc[~X['brand_name'].isin(self.pop_brands), 'brand_name'] = 'Other'
        X['category_name_l1'] = X['category_name'].str.split('/').apply(lambda x: x[0])
        X['category_name_l1s'] = \
            X['category_name'].str.split('/').apply(
                lambda x: x[0] if x[0] != 'Women' else '/'.join(x[:2]))
        X['category_name_l2'] = \
            X['category_name'].str.split('/').apply(lambda x: '/'.join(x[:2]))
        for pat, repl in self.repl_patterns:
            X['item_description'] = X['item_description'].str.replace(
                pat, repl, flags=re.IGNORECASE)

        no_description = X['item_description'] == 'No description yet'
        X.loc[no_description, 'item_description'] = ''
        X['no_description'] = no_description.astype(str)
        X['item_condition_id'] = X['item_condition_id'].map(str)
        X['shipping'] = X['shipping'].map(str)
        return X


class PreprocessDataPJ(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=4, hashchars=False, stem=True):
        self.n_jobs = n_jobs
        self.hashchars = hashchars
        self.stem = stem

    def fit(self, X, y):
        return self

    def transform(self, X):
        tokenizer = FastTokenizer()
        clean_text_ = partial(clean_text, tokenizer=tokenizer, hashchars=self.hashchars)
        X['item_condition_id'] = X['item_condition_id'].fillna('UNK').astype(str)
        X['shipping'] = X['shipping'].astype(str)
        X['item_description'][X['item_description'] == 'No description yet'] = UNK
        X['item_description'] = X['item_description'].fillna('').astype(str)
        X['name'] = X['name'].fillna('').astype(str)

        # trim
        X['item_description'] = X['item_description'].map(trim_description)
        X['name'] = X['name'].map(trim_name)
        X['brand_name'] = X['brand_name'].map(trim_brand_name)

        if self.stem:
            with Pool(4) as pool:
                X['name_clean'] = pool.map(clean_text_, tqdm(X['name'], mininterval=2), chunksize=1000)
                X['desc_clean'] = pool.map(clean_text_, tqdm(X['item_description'], mininterval=2), chunksize=1000)
                X['brand_name_clean'] = pool.map(clean_text_, tqdm(X['brand_name'], mininterval=2), chunksize=1000)
                X['category_name_clean'] = pool.map(clean_text_, tqdm(X['category_name'], mininterval=2),
                                                    chunksize=1000)
        X['no_cat'] = X['category_name'].isnull().map(int)
        cat_def = [UNK, UNK, UNK]
        X['cat_split'] = X['category_name'].fillna('/'.join(cat_def)).map(lambda x: x.split('/'))
        X['cat_1'] = X['cat_split'].map(
            lambda x: x[0] if isinstance(x, list) and len(x) >= 1 else cat_def).str.lower()
        X['cat_2'] = X['cat_split'].map(
            lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else cat_def).str.lower()
        X['cat_3'] = X['cat_split'].map(
            lambda x: x[2] if isinstance(x, list) and len(x) >= 3 else cat_def).str.lower()
        X['is_bundle'] = (X['item_description'].str.find('bundl') >= 0).map(int)

        return X


class ExtractSpecifics(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        keys = {"1", "2", "3", "4", "5", "7", "8", "9", "a", "as", "at", "b", "bars", "beautiful",
                "boots", "bottles", "bowls", "box", "boxes", "brand", "bras", "bucks",
                "cans", "card", "cards", "case", "cm", "comes",
                "compartments", "controllers", "cream", "credit", "crop", "dd",
                "dollar", "dollars", "dolls", "dress", "dvds", "each", "edition", "euc",
                "fashion", "feet", "fits", "fl", "ft", "g", "games", "gb",
                "gms", "gold", "gram", "grams",
                "hr", "hrs", "in", "inch", "inches", "k",
                "karat", "layers", "up",
                "meter", "mil", "mini", "mint", "ml", "mm", "month", "mugs", "no", "not", "nwt", "off",
                "onesies", "opi", "ounce", "ounces", "outfits", "oz", "packages", "packets", "packs", "pair", "panels",
                "pants", "patches", "pc", "pics", "piece", "pieces", "pokémon",
                "pokemon", "pounds", "price", "protection", "random", "retro", "ring", "rings", "rolls",
                "samples", "sandals", "series", "sets", "sheets", "shirts", "shoe", "shoes",
                "shows", "slots", "small", "so", "some", "stamped", "sterling", "stickers", "still", "stretch",
                "strips", "summer", "t", "tags", "tiny", "tone", "tubes", "victoria", "vinyl", "w", "waist",
                "waistband", "waterproof", "watt", "white", "wireless", "x10", "x13", "x15", "x3", "x4", "x5", "x6",
                "x7", "x8", "x9", "yrs", "½", "lipsticks", "bar", "apple", "access", "wax", "monster", "spell",
                "spinners", "lunch", "ac", "jamberry", "medal", "gerard"}
        regex = re.compile("(\d+)[ ]?(\w+)", re.IGNORECASE)

        specifics = []
        for x in X:
            spec = {}
            for val, key in regex.findall(str(x), re.IGNORECASE):
                if key in keys:
                    val = try_float(val)
                    if val > 3000:
                        continue
                    spec[key] = val
                    spec['{}_{}'.format(key, val)] = 1
            specifics.append(spec)

        return specifics


def _make_float_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("f"))


class PredictProbaTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, est, target_column):
        self.target_column = target_column
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, X[self.target_column])
        return self

    def transform(self, X):
        return self.est.predict_proba(X)


class NumericalVectorizer(CountVectorizer):
    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = _make_int_array()
        values = _make_float_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            current_num = 0
            for feature in analyze(doc):
                maybe_float = try_float(feature)
                if maybe_float > 0 and maybe_float <= 200:
                    current_num = maybe_float
                    continue
                try:
                    if current_num == 0:
                        continue
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = current_num / 200
                        current_num = 0
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.float32)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=np.float32)
        X.sort_indices()
        return vocabulary, X


class SanitizeSparseMatrix(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        self.datamax = np.nanmax(X.data)
        return self

    def transform(self, X):
        X.data[np.isnan(X.data)] = 0
        X.data = X.data.clip(0, self.datamax)
        return X
