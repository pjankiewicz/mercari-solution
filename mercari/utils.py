from functools import wraps
import time

import psutil
import numpy as np

from mercari.config import logger


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        logger.info('Starting {}'.format(self.message))
        self.start_clock = time.clock()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_clock = time.clock()
        self.end_time = time.time()
        self.interval_clock = self.end_clock - self.start_clock
        self.interval_time = self.end_time - self.start_time
        logger.info('Finished {}. Took {:.2f} seconds, CPU time {:.2f}, effectiveness {:.2f}'.format(
            self.message, self.interval_time, self.interval_clock, self.interval_clock / self.interval_time))


def try_float(t):
    try:
        return float(t)
    except:
        return 0


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def log_time(fn, name):
    @wraps(fn)
    def deco(*args, **kwargs):
        logger.info(f'[{name}] << starting {fn.__name__}')
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.time() - t0
            logger.info(f'[{name}] >> finished {fn.__name__} in {dt:.2f} s, '
                         f'{memory_info()}')
    return deco


def memory_info():
    process = psutil.Process()
    memory_info = process.memory_info()
    return (f'process {process.pid}: RSS {memory_info.rss:,} {memory_info}; '
            f'system: {psutil.virtual_memory()}')
