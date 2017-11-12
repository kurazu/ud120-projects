import time


class Timer:
    start = 0
    end = 0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.end = time.time()

    def __call__(self):
        return self.end - self.start
