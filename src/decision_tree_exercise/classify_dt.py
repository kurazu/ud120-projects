from sklearn.tree import DecisionTreeClassifier


def classify(features_train, labels_train):
    clf = DecisionTreeClassifier(min_samples_split=2)
    clf.fit(features_train, labels_train)
    return clf
