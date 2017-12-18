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
    features = ["salary", "bonus"]
    data = featureFormat(data_dict, features)
    ### your code below


if __name__ == '__main__':
    main()
