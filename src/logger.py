import logging
from typing import Optional, Union, Dict


# Save already initialized loggers
INITIALIZED_LOGGERS: Dict[str, bool] = {}


def get_logger(name: str,
               log_file: Optional[str] = None,
               log_level: Union[int, str] = logging.INFO,
               file_mode: str = 'w') -> logging.Logger:
    """
    Get a logger with the given name. If the logger has already been
    initialized, return the existing logger.

    Args:
        name (str): name of logger
        log_file (str, optional): file to save logs. If None - only console.
        log_level (int or str): level of the logger
        file_mode (str): mode to open the logfile

    Returns:
        logging.Logger: logger
    """

    # Check logs level valug
    if log_level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
        raise ValueError("Invalid log level")
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    logger = logging.getLogger(name)

    if name in INITIALIZED_LOGGERS:
        return logger

    for logger_name in INITIALIZED_LOGGERS:
        if name.startswith(logger_name):
            return logger

    # Set root handler to ERROR to fix dublicates
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    INITIALIZED_LOGGERS[name] = True

    return logger
