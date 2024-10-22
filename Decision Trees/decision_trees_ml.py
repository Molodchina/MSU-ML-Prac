import os
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        self.shape = x.shape[1:]
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        x = self.center_potentials(x)
        return x.reshape((x.shape[0], -1))

    def center_potentials(self, x):
        """
        Center potentials in the image.
        :param x: list of potential's 2d matrices
        :return: centered potentials
        """
        x_centered = x.copy()
        for i in range(x.shape[0]):
            potential = x[i]
            center = potential.shape[0] // 2, potential.shape[1] // 2
            dx, dy = np.unravel_index(potential.argmin(), potential.shape)
            shift = (center[0] - dx, center[1] - dy)
            x_centered[i] = np.roll(np.roll(potential, shift[0], axis=0), shift[1], axis=1)
        return x_centered


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    regressor = Pipeline([
        ('vectorizer', PotentialTransformer()),
        ('ensemble', ExtraTreesRegressor(n_estimators=250))
    ])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
