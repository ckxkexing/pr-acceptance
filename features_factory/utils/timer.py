######################################
# 计时器
######################################

import time


class Timer:
    def __init__(self):
        self.time = time.time()

    def start(self):
        self.time = time.time()

    def stop(self):
        return time.time() - self.time


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        timer = Timer()
        start_time = timer.start()
        result = func(*args, **kwargs)
        end_time = timer.stop()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.2f} seconds.")
        return result

    return wrapper
