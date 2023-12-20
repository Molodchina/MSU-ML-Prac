import numpy as np
import typing


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x):
        return (x - self.mean) / self.std


class MinMaxScaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, x):
        self.min_val = np.min(x, axis=0)
        self.max_val = np.max(x, axis=0)

    def transform(self, x):
        return (x - self.min_val) / (self.max_val - self.min_val)
