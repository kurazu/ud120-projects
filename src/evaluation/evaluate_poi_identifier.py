"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os.path

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from tools.feature_format import featureFormat, targetFeatureSplit
from tools.loading import load_pickle

import final_project
import tools


def main():
    keys_path = os.path.join(
        os.path.dirname(tools.__file__),
        'python2_lesson14_keys.pkl'
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

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    # it's all yours from here forward!
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    print(clf.score(features_test, labels_test))


if __name__ == '__main__':
    main()
