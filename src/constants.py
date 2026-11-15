"""Project-wide constants and structured logging configuration.

Defines deterministic random seeds and standardized ISO 8601 logging
configuration shared across all FleetReliability-AI modules.
"""

import logging

RANDOM_STATE: int = 42
LOGGER_NAME: str = "fleetreliability"
LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
LOG_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures and returns the standardized project logger.

    Sets up structured ISO 8601 logging with module, function, and line number
    context for the ``fleetreliability`` logger without adding duplicate
    handlers across repeated calls.

    Args:
        level: Minimum logging severity level (for example, ``logging.INFO`` or
            ``logging.DEBUG``). Defaults to ``logging.INFO``.

    Returns:
        logging.Logger: Configured logger instance named ``LOGGER_NAME``.

    Raises:
        ValueError: If ``level`` is not a valid logging level recognized by
            ``logging.Logger.setLevel``.

    Example:
        >>> import logging
        >>> logger = configure_logging(logging.INFO)
        >>> logger.name
        'fleetreliability'
    """
    logger: logging.Logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    formatter: logging.Formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    if not logger.handlers:
        handler: logging.StreamHandler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for existing_handler in logger.handlers:
            existing_handler.setLevel(level)
            existing_handler.setFormatter(formatter)

    return logger
