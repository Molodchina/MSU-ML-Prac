from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    i, j = 0, 0
    diag = []
    while i < min(len(X), len(X[0])):
        diag.append(X[i][i])
        i += 1
    return -1 if all(x < 0 for x in diag) \
        else sum((x > 0) * x for x in diag)


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    if len(x) != len(y):
        return False

    a = {}
    b = {}
    for i in range(len(x)):
        a[x[i]] = a.get(x[i], 0) + 1
        b[y[i]] = b.get(y[i], 0) + 1

    return a == b


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if len(x) < 2:
        return -1

    return max(a * b if ((a % 3 == 0) | (b % 3 == 0)) else -1 for a, b in zip(x, x[1:]))


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    return [[sum([(image[i][j][k]) * weights[k]
                  for k in range(len(weights))])
             for j in range(len(image[0]))]
            for i in range(len(image))]


def rle_scalar(x: List[List[int]], y: List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    z = sum(n for _, n in x)
    if z != sum(n for _, n in y):
        return -1
    res1 = ''
    res2 = ''
    for a, n in x:
        res1 += str(a) * n
    for a, n in y:
        res2 += str(a) * n
    return sum(int(res1[i]) * int(res2[i]) for i in range(z))


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    n1 = [sum([y ** 2 for y in z]) ** 0.5 for z in X]
    n2 = [sum([y ** 2 for y in z]) ** 0.5 for z in Y]
    res = [[] for x in range(len(X))]
    for x in range(len(X)):
        for y in range(len(Y)):
            s = sum([X[x][i] * Y[y][i] for i in range(len(X[0]))])
            res[x].append(1. if (n1[x] * n2[y]) == 0
                          else s / (n1[x] * n2[y]))
    return res
