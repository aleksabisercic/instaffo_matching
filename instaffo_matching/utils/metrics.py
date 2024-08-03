import time
import logging

logger = logging.getLogger(__name__)

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper