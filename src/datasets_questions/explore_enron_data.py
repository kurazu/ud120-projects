"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import os.path
import pickle
import final_project

THERE = os.path.dirname(final_project.__file__)
DATASET_PATH = os.path.join(THERE, 'final_project_dataset.pkl')


def main():
    with open(DATASET_PATH, 'rb') as f:
        data = pickle.load(f)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
