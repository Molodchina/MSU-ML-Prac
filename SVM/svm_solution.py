import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    estimators = [('lr', LogisticRegression()), ('dt', DecisionTreeClassifier())]
    clf = StackingClassifier(estimators=estimators, final_estimator=SVC(C=25.118864315095795, kernel='rbf'))
    clf.fit(train_features, train_target)

    return clf.predict(test_features)
