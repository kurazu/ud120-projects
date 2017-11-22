#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

from tools.email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from tools import timing, confusion


def main():
    features_train, features_test, labels_train, labels_test = preprocess()

    # features_train = features_train[:len(features_train) // 100]
    # labels_train = labels_train[:len(labels_train) // 100]

    clf = DecisionTreeClassifier(min_samples_split=40)
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
    print(
        'Confusion matrix',
        confusion.confusion_matrix(
            labels_test, predictions, neg_label='Sara', pos_label='Chris'
        )
    )


if __name__ == '__main__':
    main()
