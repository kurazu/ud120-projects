#!/usr/bin/python

"""
    Starter code for the regression mini-project.

    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""


import pickle
import io
import os.path

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

import tools
from tools.feature_format import featureFormat, targetFeatureSplit
import final_project



def main():
    dictionary_path = os.path.join(
        os.path.dirname(final_project.__file__),
        "final_project_dataset_modified.pkl"
    )
    with io.open(dictionary_path, 'rb') as f:
        dictionary = pickle.load(f)

    ### list the features you want to look at--first item in the
    ### list will be the "target" feature
    features_list = ["bonus", "salary"]
    keys_path = os.path.join(
        os.path.dirname(tools.__file__),
        'python2_lesson06_keys.pkl'
    )
    data = featureFormat(
        dictionary, features_list, remove_any_zeroes=True, sort_keys=keys_path
    )
    target, features = targetFeatureSplit(data)

    ### training-testing split needed in regression, just like classification
    feature_train, feature_test, target_train, target_test = train_test_split(
        features, target, test_size=0.5, random_state=42
    )
    train_color = "b"
    test_color = "r"



    ### Your regression goes here!
    ### Please name it reg, so that the plotting code below picks it up and
    ### plots it correctly. Don't forget to change the test_color above from "b" to
    ### "r" to differentiate training points from test points.


    reg = LinearRegression()
    reg.fit(feature_train, target_train)

    print('test score', reg.score(feature_test, target_test))
    print('train score', reg.score(feature_train, target_train))

    print('coef', reg.coef_, 'intercept', reg.intercept_)

    ### draw the scatterplot, with color-coded training and testing points

    for feature, target in zip(feature_test, target_test):
        plt.scatter(feature, target, color=test_color)
    for feature, target in zip(feature_train, target_train):
        plt.scatter(feature, target, color=train_color)

    ### labels for the legend
    plt.scatter(
        feature_test[0], target_test[0], color=test_color, label="test"
    )
    plt.scatter(
        feature_test[0], target_test[0], color=train_color, label="train"
    )




    ### draw the regression line, once it's coded
    try:
        plt.plot(feature_test, reg.predict(feature_test))
    except NameError:
        pass
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[0])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
