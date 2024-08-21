import numpy as np


def gini(sample):
    _, counts = np.unique(sample, return_counts=True)
    probas = counts / counts.sum()
    return 1 - np.sum(probas ** 2)


def entropy(sample):
    _, counts = np.unique(sample, return_counts=True)
    probas = counts / counts.sum()
    return -np.sum(probas * np.log(probas))


def classification_error(sample):
    _, counts = np.unique(sample, return_counts=True)
    return 1 - counts.max() / counts.sum()


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    measures = {'gini': float(gini(sample)),
                'entropy': float(entropy(sample)),
                'error': float(classification_error(sample))}
    return measures
