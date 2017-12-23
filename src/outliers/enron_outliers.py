#!/usr/bin/python

import pickle
import matplotlib.pyplot

from tools.feature_format import featureFormat, targetFeatureSplit
from tools.loading import load_pickle

import final_project
import os.path


def main():
    ### read in data dictionary, convert to numpy array
    file_path = os.path.join(
        os.path.dirname(final_project.__file__), 'final_project_dataset.pkl'
    )
    data_dict = load_pickle(file_path)
    data_dict.pop('TOTAL')
    features = ["salary", "bonus"]
    data = featureFormat(data_dict, features)
    ### your code below
    for salary, bonus in data:
        matplotlib.pyplot.scatter(salary, bonus)

    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.show()

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
