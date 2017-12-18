import io
import pickle


def load_pickle(path):
    with io.open(path, 'rb') as f:
        return pickle.load(f)
