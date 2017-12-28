import numpy
from operator import itemgetter

import text_learning
from tools.loading import load_pickle
import os.path

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    numpy.random.seed(42)
    # The words (features) and authors (labels), already largely processed.
    # These files should have been created from the previous (Lesson 10)
    # mini-project.
    word_data = load_pickle(os.path.join(
        os.path.dirname(text_learning.__file__),
        'your_word_data.pkl'
    ))
    authors = load_pickle(os.path.join(
        os.path.dirname(text_learning.__file__),
        'your_email_authors.pkl'
    ))

    # test_size is the percentage of events assigned to the test set (the
    # remainder go into training)
    # feature matrices changed to dense representations for compatibility with
    # classifier functions in versions 0.15.2 and earlier
    features_train, features_test, labels_train, labels_test = \
        cross_validation.train_test_split(
            word_data, authors, test_size=0.1, random_state=42
        )

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train = vectorizer.fit_transform(features_train)
    features_test  = vectorizer.transform(features_test).toarray()


    # a classic way to overfit is to use a small number
    # of data points and a large number of features;
    # train on only 150 events to put ourselves in this regime
    features_train = features_train[:150].toarray()
    labels_train   = labels_train[:150]

    # your code goes here
    classifier = DecisionTreeClassifier()
    classifier.fit(features_train, labels_train)

    train_predictions = classifier.predict(features_train)
    print('Train score', accuracy_score(labels_train, train_predictions))

    test_predictions = classifier.predict(features_test)
    print('Test score', accuracy_score(labels_test, test_predictions))

    sorted_features = sorted(
        enumerate(classifier.feature_importances_),
        key=itemgetter(1),
        reverse=True
    )
    for idx, feature_importance in sorted_features[:20]:
        print(feature_importance, idx)


if __name__ == '__main__':
    main()
