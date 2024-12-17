import time
from typing import Callable


class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    PURPLE = "\033[0;35m"
    LIGHT_GRAY = "\033[0;37m"
    LIGHT_RED = "\033[1;31m"
    END = "\033[0m"

def log_elapsed_time(func: Callable):
    """Decorator for timing a function."""

    def wrapper_log_elapsed_time(*args, **kwargs):
        print(f"{func.__name__}({args}, {kwargs}) {Colors.LIGHT_RED}...{Colors.END}")
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"{Colors.LIGHT_GRAY}Elapsed Time: {Colors.PURPLE}{round(end - start, 2)}{Colors.END}s\n")
        return res

    return wrapper_log_elapsed_time