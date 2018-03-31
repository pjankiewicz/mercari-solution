import pandas as pd
from sklearn.model_selection import train_test_split

from mercari.config import DEBUG, DEBUG_N, TEST_SIZE, TEST_CHUNK, VALIDATION_SIZE


def load_train(path='../input/train.tsv'):
    if DEBUG:
        return pd.read_csv(path, sep='\t').query('price > 0').iloc[:DEBUG_N, :]
    else:
        return pd.read_csv(path, sep='\t').query('price > 0')


def load_train_validation():
    return mercari_train_test_split(load_train())


def load_test_iter():
    for _ in range(TEST_SIZE):
        for df in pd.read_csv('../input/test.tsv', sep='\t', chunksize=TEST_CHUNK):
            if DEBUG:
                yield df.iloc[:DEBUG_N]
            else:
                yield df


def mercari_train_test_split(*arrays):
    return train_test_split(*arrays, test_size=VALIDATION_SIZE, random_state=0)
