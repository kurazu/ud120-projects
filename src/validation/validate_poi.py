#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import os.path

from sklearn.tree import DecisionTreeClassifier

from tools.feature_format import featureFormat, targetFeatureSplit
from tools.loading import load_pickle

import final_project
import tools


def main():
    keys_path = os.path.join(
        os.path.dirname(tools.__file__),
        'python2_lesson13_keys.pkl'
    )

    data_dict = load_pickle(os.path.join(
        os.path.dirname(final_project.__file__), 'final_project_dataset.pkl'
    ))

    # first element is our labels, any added elements are predictor
    # features. Keep this the same for the mini-project, but you'll
    # have a different feature list when you do the final project.
    features_list = ["poi", "salary"]

    data = featureFormat(data_dict, features_list, sort_keys=keys_path)
    labels, features = targetFeatureSplit(data)

    # it's all yours from here forward!
    clf = DecisionTreeClassifier()
    clf.fit(features, labels)
    print(clf.score(features, labels))


if __name__ == '__main__':
    main()
