import time
import logging
from functools import wraps

# Setting up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ExecutionTimer:
    def __init__(self, func):
        self.func = func
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(
            f"Execution time of {self.func.__name__}: {execution_time:.6f} seconds"
        )
        return result

    @classmethod
    def time_this(cls, func):
        return cls(func)
