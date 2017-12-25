#!/usr/bin/python

import os.path
import re
import io

from tools.parse_out_email_text import parseOutText
from tools.loading import dump_pickle

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

HERE = os.path.dirname(__file__)
TOP_DIR = os.path.join(HERE, '..', '..')


def process(from_sara, from_chris):
    from_data = []
    word_data = []

    for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
        for path in from_person:
            path = os.path.join(TOP_DIR, path[:-1])
            print(path)
            with io.open(path, 'r', encoding='utf-8') as email:
                pass

            # use parseOutText to extract the text from the opened email

            # use str.replace() to remove any instances of the words
            # ["sara", "shackleton", "chris", "germani"]

            # append the text to word_data

            # append a 0 to from_data if email is from Sara,
            # and 1 if email is from Chris

    print("emails processed")

    dump_pickle(os.path.join(HERE, "your_word_data.pkl"), word_data)
    dump_pickle(os.path.join(HERE, "your_email_authors.pkl"), from_data)

    # in Part 4, do TfIdf vectorization here


def main():
    sara_path = os.path.join(HERE, 'from_sara.txt')
    chris_path = os.path.join(HERE, 'from_chris.txt')
    with \
        io.open(sara_path, 'r', encoding='utf-8') as from_sara, \
        io.open(chris_path, 'r', encoding='utf-8') as from_chris \
    :
        process(from_sara, from_chris)


if __name__ == '__main__':
    main()
