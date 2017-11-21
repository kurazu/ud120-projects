#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

from tools.email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn import metrics
from tools import timing, confusion


def main():
    features_train, features_test, labels_train, labels_test = preprocess()

    # features_train = features_train[:len(features_train) // 100]
    # labels_train = labels_train[:len(labels_train) // 100]

    for C in [10000]:
        clf = SVC(kernel='rbf', C=C)
        timer = timing.Timer()
        with timer:
            clf.fit(features_train, labels_train)
        print('C', C, 'fit took', timer())
        with timer:
            predictions = clf.predict(features_test)
        print('C', C, 'predict took', timer())
        with timer:
            score = metrics.accuracy_score(labels_test, predictions)
        print('C', C, 'score took', timer())
        print('C', C, 'SCORE', score)
        print(
            'Confusion matrix',
            confusion.confusion_matrix(
                labels_test, predictions, neg_label='Sara', pos_label='Chris'
            )
        )


if __name__ == '__main__':
    main()
