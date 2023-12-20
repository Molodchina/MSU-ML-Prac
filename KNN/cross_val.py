import numpy as np
from collections import defaultdict
import typing


def kfold_split(n, k):
    fold_size = n // k
    indices = np.arange(n)
    splits = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        if i == k - 1:
            end = n
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))
        splits.append((train_indices, val_indices))
    return splits


def knn_cv_score(x, y, parameters, score_fun, folds, regressor):
    results = {}
    for train_index, val_index in folds:
        X_train, X_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        for normalizer, normalizer_name in parameters["normalizers"]:
            if normalizer is not None:
                normalizer.fit(X_train)
                X_train = normalizer.transform(X_train)
                X_val = normalizer.transform(X_val)

            for neighbor in parameters["n_neighbors"]:
                for metric in parameters["metrics"]:
                    for weight in parameters["weights"]:
                        model = regressor(n_neighbors=neighbor, metric=metric, weights=weight)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = score_fun(y_val, y_pred)
                        key = (normalizer_name, neighbor, metric, weight)
                        if key not in results:
                            results[key] = []
                        results[key].append(score)
    for x in results:
        results[x] = np.mean(results[x])
    return results
