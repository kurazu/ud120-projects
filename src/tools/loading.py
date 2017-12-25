import io
import pickle


def load_pickle(path):
    with io.open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(path, obj):
    with io.open(path, 'wb') as f:
        pickle.dump(obj, f)
