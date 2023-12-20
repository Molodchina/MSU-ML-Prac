import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    diag = np.diagonal(X)
    return np.sum(-1 if np.all(diag < 0) else sum((diag >= 0) * diag))


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    if x.shape != y.shape:
        return False
    return np.all(np.sort(x) == np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if x.size < 2:
        return -1
    x = np.append(x, x[-2])
    return np.where(x[:-1] % 3 == 0, x[:-1] * x[1:], -1).max()


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.sum(np.multiply(image, weights), axis=2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    if np.sum(x, axis=0)[1] != np.sum(y, axis=0)[1]:
        return -1
    return np.dot(np.repeat(x[:, 0], x[:, 1]),
                  np.repeat(y[:, 0], y[:, 1]))


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    res = np.ones(shape=(len(X), len(Y)))
    normX = np.linalg.norm(X, axis=1)[:, np.newaxis]
    normY = np.linalg.norm(Y, axis=1)[:, np.newaxis]
    res[:] = np.divide(X[:] @ Y[:].T, b := normX[:] @ normY[:].T, out=np.ones_like(res), where=b!=0)
    return res
