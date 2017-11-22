import numpy
import os.path
import matplotlib.pyplot as plt

from terrain.prep_terrain_data import makeTerrainData
from terrain.class_vis import prettyPicture

from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from tools import timing, confusion
from sklearn.model_selection import GridSearchCV

HERE = os.path.dirname(__file__)


def main():
    features_train, labels_train, features_test, labels_test = \
        makeTerrainData()

    # the training data (features_train, labels_train)
    # have both "fast" and "slow" points mixed
    # in together--separate them so we can give them different
    # colors in the scatterplot,
    # and visually identify them

    grade_fast = [
        features_train[ii][0]
        for ii in range(0, len(features_train))
        if labels_train[ii] == 0
    ]
    bumpy_fast = [
        features_train[ii][1]
        for ii in range(0, len(features_train))
        if labels_train[ii] == 0
    ]
    grade_slow = [
        features_train[ii][0]
        for ii in range(0, len(features_train))
        if labels_train[ii] == 1
    ]
    bumpy_slow = [
        features_train[ii][1]
        for ii in range(0, len(features_train))
        if labels_train[ii] == 1
    ]

    # initial visualization
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
    plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()
    # /initial visualization

    param_grid = {
        'n_estimators': list(range(5, 200 + 1, 5)),
        'algorithm': ['SAMME', 'SAMME.R'],
        'learning_rate': numpy.arange(0.25, 2.25, 0.25)
    }

    clf = AdaBoostClassifier(random_state=1)
    scorer = metrics.make_scorer(metrics.accuracy_score)
    grid_search = GridSearchCV(
        clf, param_grid=param_grid,
        scoring=scorer, verbose=2, n_jobs=-1,
    )

    print('Optimizing params ...')
    grid_search.fit(X=features_train, y=labels_train)
    print('Optimization finished')
    print('Results', grid_search.cv_results_)
    best_pipeline = grid_search.best_estimator_
    best_params = best_pipeline.get_params()
    print('Best params', best_params)
    best_model = best_pipeline

    predictions = best_model.predict(features_test)
    best_score = metrics.accuracy_score(predictions, labels_test)
    print('BEST score', best_score)
    # draw the decision boundary with the text points overlaid
    prettyPicture(HERE, best_model, features_test, labels_test)


if __name__ == '__main__':
    main()
