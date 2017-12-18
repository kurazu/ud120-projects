#!/usr/bin/python

import os.path
import random
import numpy
import matplotlib.pyplot as plt

from .outlier_cleaner import outlierCleaner
from tools.loading import load_pickle
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


HERE = os.path.dirname(__file__)


def main():
    ### load up some practice data with outliers in it
    ages = load_pickle(os.path.join(HERE, 'practice_outliers_ages.pkl'))
    net_worths = load_pickle(os.path.join(HERE, 'practice_outliers_net_worths.pkl'))


    ### ages and net_worths need to be reshaped into 2D numpy arrays
    ### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
    ### by convention, n_rows is the number of data points
    ### and n_columns is the number of features
    ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))
    ages_train, ages_test, net_worths_train, net_worths_test = \
        train_test_split(ages, net_worths, test_size=0.1, random_state=42)

    ### fill in a regression here!  Name the regression object reg so that
    ### the plotting code below works, and you can see what your regression looks like
    reg = LinearRegression()
    reg.fit(ages_train, net_worths_train)
    print('slope', reg.coef_, 'intercept', reg.intercept_)
    print('score', reg.score(ages_test, net_worths_test))

    plt.plot(ages, reg.predict(ages), color="blue")
    plt.scatter(ages, net_worths)
    plt.show()


    ### identify and remove the most outlier-y points
    cleaned_data = []
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner(
        predictions, ages_train, net_worths_train
    )


    # Refit
    ages, net_worths, errors = list(zip(*cleaned_data))
    ages       = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    reg.fit(ages, net_worths)
    print('slope', reg.coef_, 'intercept', reg.intercept_)
    print('score', reg.score(ages_test, net_worths_test))
    plt.plot(ages, reg.predict(ages), color="blue")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()



if __name__ == '__main__':
    main()
