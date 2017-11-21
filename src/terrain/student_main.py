""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.

    The objective of this exercise is to recreate the decision
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """

import os.path
from .prep_terrain_data import makeTerrainData
from .class_vis import prettyPicture
from .classify import classify

# import numpy as np
# import pylab as pl

HERE = os.path.dirname(__file__)


def main():
    features_train, labels_train, features_test, labels_test = \
        makeTerrainData()

    # the training data (features_train, labels_train)
    # have both "fast" and "slow" points mixed
    # in together--separate them so we can give them different
    # colors in the scatterplot,
    # and visually identify them

    # grade_fast = [
    #     features_train[ii][0]
    #     for ii in range(0, len(features_train))
    #     if labels_train[ii] == 0
    # ]
    # bumpy_fast = [
    #     features_train[ii][1]
    #     for ii in range(0, len(features_train))
    #     if labels_train[ii] == 0
    # ]
    # grade_slow = [
    #     features_train[ii][0]
    #     for ii in range(0, len(features_train))
    #     if labels_train[ii] == 1
    # ]
    # bumpy_slow = [
    #     features_train[ii][1]
    #     for ii in range(0, len(features_train))
    #     if labels_train[ii] == 1
    # ]

    clf = classify(features_train, labels_train)

    # draw the decision boundary with the text points overlaid
    prettyPicture(HERE, clf, features_test, labels_test)


if __name__ == '__main__':
    main()
