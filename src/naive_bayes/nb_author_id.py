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


def main():
    features_train, features_test, labels_train, labels_test = preprocess()
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    print('SCORE', clf.score(features_test, labels_test))
    pass


if __name__ == '__main__':
    main()
