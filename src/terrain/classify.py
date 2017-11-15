from sklearn.svm import SVC


def classify(features_train, labels_train):
    classifier = SVC(kernel='rbf', C=100000)
    classifier.fit(features_train, labels_train)
    return classifier
