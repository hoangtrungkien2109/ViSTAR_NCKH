"""Class to measure processing time"""
import time
from loguru import logger

def processing_time(func):
    """Measure the processing time of a function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the function
        logger.info(f"Processing time for {func.__name__}: {(time.time() - start_time):.4f} seconds.")
        return result  # Return the result of the function
    return wrapper
