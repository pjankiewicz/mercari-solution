import numpy as np


def binarize(ys, percentiles, soft=True, soft_scale=160):
    if soft:
        mean_percentiles = get_mean_percentiles(percentiles)
        binarized = np.exp(-0.5 * soft_scale * (
                (np.expand_dims(ys, 1) - mean_percentiles) ** 2))
        binarized = (binarized.T / binarized.sum(axis=1)).T
    else:
        binarized = np.zeros((len(ys), len(percentiles) - 1), dtype=np.float32)
        for i in range(1, len(percentiles)):
            binarized[:, i - 1] = (
                    (ys > percentiles[i - 1]) & (ys <= percentiles[i]))
    return binarized


def get_percentiles(ys, n_bins):
    return np.percentile(ys, list(np.arange(0, 100, 100 / n_bins)) + [100])


def get_mean_percentiles(percentiles):
    return np.mean([percentiles[:-1], percentiles[1:]], axis=0)
