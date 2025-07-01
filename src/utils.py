import os
import logging
from loguru import logger

def setup_logging(log_file: str = None):
    """
    Configure logging via loguru and standard logging.
    If log_file is provided, logs are written to file as well.
    """
    # Remove default handlers
    logger.remove()
    # Add stdout
    logger.add(lambda msg: print(msg, end=''), level="INFO")
    if log_file:
        # Rotating log: max 10 MB, keep 3 backups
        logger.add(log_file, rotation="10 MB", retention=3)
    # Also configure standard logging to forward to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except Exception:
                level = record.levelno
            # Find caller from where originated
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
