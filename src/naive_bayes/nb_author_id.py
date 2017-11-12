#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

from tools.email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from tools import timing


def main():
    features_train, features_test, labels_train, labels_test = preprocess()
    clf = GaussianNB()
    timer = timing.Timer()
    with timer:
        clf.fit(features_train, labels_train)
    print('fit took', timer())
    with timer:
        predictions = clf.predict(features_test)
    print('predict took', timer())
    with timer:
        score = metrics.accuracy_score(labels_test, predictions)
    print('score took', timer())
    print('SCORE', score)


if __name__ == '__main__':
    main()
