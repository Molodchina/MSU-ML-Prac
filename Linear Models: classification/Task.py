import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.n_features = 0
        self.categories = []
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.categories = []
        self.n_features = X.shape[1]
        for i in range(self.n_features):
            self.categories.append(sorted(X.iloc[:, i].unique()))

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        res = []
        for i in range(self.n_features):
            cat = self.categories[i]
            col = X.iloc[:, i]
            one_hot = np.zeros((len(col), len(cat)))
            for j in range(len(cat)):
                one_hot[:, j] = (col == cat[j])
            res.append(one_hot)
        return np.concatenate(res, axis=1)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.successes = np.array([], dtype=dict)
        self.counters = np.array([], dtype=dict)
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        x = np.array(X)
        y = np.array(Y)

        for i in range(x.shape[1]):
            uniques = np.unique(x[:, i])
            success_dict = {}
            counter_dict = {}

            for unique in uniques:
                mask = x[:, i] == unique
                success_dict[unique] = np.sum(mask * y) / np.sum(mask)
                counter_dict[unique] = np.sum(mask) / len(y)

            self.successes = np.append(self.successes, success_dict)
            self.counters = np.append(self.counters, counter_dict)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        x = np.array(X)
        transformed = np.zeros((x.shape[0], 3 * x.shape[1]))

        for i in range(x.shape[0]):
            values = []
            for s, c, val in zip(self.successes, self.counters, x[i]):
                values += [s[val], c[val], (s[val] + a) / (c[val] + b)]
            transformed[i] = np.array(values)
        return transformed

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        # your code here

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        # your code here

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    # применяем one-hot кодирование к x
    learning_rate = 0.1
    n_iterations = 1000
    x_one_hot = np.eye(len(np.unique(x)))[x]
    # инициализируем веса случайными значениями
    w = np.random.rand(x_one_hot.shape[1])
    # запускаем градиентный спуск
    for i in range(n_iterations):
        # вычисляем прогнозы
        p = np.dot(x_one_hot, w)
        # вычисляем градиент
        grad = 1 / len(y) * np.dot((p - y), x_one_hot)
        # обновляем веса
        w = w - learning_rate * grad
    return w.tolist()
