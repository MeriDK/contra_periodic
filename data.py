import numpy as np
from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.z[i]


def times_to_lags(x, p=None):
    lags = x[:, 1:] - x[:, :-1]
    if p is not None:
        lags = np.c_[lags, x[:, 0] - x[:, -1] + p]
    return lags


def preprocess(X_raw, periods, use_error=False):
    N, L, F = X_raw.shape
    out_dim = 3 if use_error else 2
    X = np.zeros((N, L, out_dim))
    # TODO Check later why we don't use times_to_lags
    # X[:, :, 0] = times_to_lags(X_raw[:, :, 0], periods) / periods[:, None]
    X[:, :, 0] = X_raw[:, :, 0]
    X[:, :, 1:out_dim] = X_raw[:, :, 1:out_dim]
    means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
    scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
    X[:, :, 1] -= means
    X[:, :, 1] /= scales
    return X, means, scales


def train_test_split(y, train_size=0.33, random_state=0):
    if random_state != -1:
        np.random.seed(random_state)
    labels = np.copy(y)
    np.random.shuffle(labels)
    y_unique = np.unique(y)
    indexes = np.arange(len(y))
    x_split = [np.array(indexes[y == label]) for label in y_unique]
    for i in range(len(y_unique)):
        if random_state != -1:
            np.random.shuffle(x_split[i])
    trains = [x for el in x_split for x in el[:max(int(train_size * len(el) + 0.5), 1)]]
    tests = [x for el in x_split for x in el[max(int(train_size * len(el) + 0.5), 1):]]
    return trains, tests


# allow random cyclic permutation on the fly
# as data augmentation for the non-invariant networks
def permute(x):
    seq_length = x.shape[2]
    for i in range(x.shape[0]):
        start = np.random.randint(0, seq_length - 1)
        x[i] = torch.cat((x[i, :, start:], x[i, :, :start]), dim=1)
    return x
