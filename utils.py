import time
from typing import Callable

def log_elapsed_time(func: Callable):
    """Decorator for timing a function."""

    def wrapper_log_elapsed_time(*args, **kwargs):
        print("/" * 10)
        print(f"RUNNING {func.__name__}({args}, {kwargs})")
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"ELAPSED TIME FOR {func.__name__}: {round(end - start, 2)}\n")
        return res

    return wrapper_log_elapsed_time