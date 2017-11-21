import os.path

from terrain.class_vis import prettyPicture
from terrain.prep_terrain_data import makeTerrainData

from .classify_dt import classify

from sklearn import metrics


HERE = os.path.dirname(__file__)


def main():
    (
        features_train, labels_train, features_test, labels_test
    ) = makeTerrainData()
    clf = classify(features_train, labels_train)
    prettyPicture(HERE, clf, features_test, labels_test)
    predictions = clf.predict(features_test)
    print('ACCURACY', metrics.accuracy_score(labels_test, predictions))


if __name__ == '__main__':
    main()
