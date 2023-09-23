import json

import numpy as np
from torch.utils.data import Dataset
import torch
import joblib


class MyDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.z[i]


def load_data(args):
    x_train, aux_train, label_train = joblib.load(f'data/{args.dataset}/train.pkl')
    x_val, aux_val, label_val = joblib.load(f'data/{args.dataset}/val.pkl')
    x_test, aux_test, label_test = joblib.load(f'data/{args.dataset}/test.pkl')

    train_dataset = MyDataset(x_train, aux_train, label_train)
    val_dataset = MyDataset(x_val, aux_val, label_val)
    test_dataset = MyDataset(x_test, aux_test, label_test)

    scales_all = np.load(f'data/{args.dataset}/scales.npy')

    with open(f'data/{args.dataset}/info.json', 'r') as f:
        info = json.loads(f.read())

    return train_dataset, val_dataset, test_dataset, scales_all, info


def times_to_lags(x, p=None):
    lags = x[:, 1:] - x[:, :-1]
    if p is not None:
        lags = np.c_[lags, x[:, 0] - x[:, -1] + p]
    return lags


# allow random cyclic permutation on the fly
# as data augmentation for the non-invariant networks
def permute(x):
    seq_length = x.shape[2]
    for i in range(x.shape[0]):
        start = np.random.randint(0, seq_length - 1)
        x[i] = torch.cat((x[i, :, start:], x[i, :, :start]), dim=1)
    return x
