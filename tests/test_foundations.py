"""Unit tests for transversal foundations (constants, logging, exceptions)."""

import io
import logging
import re

import pytest

from src.constants import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOGGER_NAME,
    RANDOM_STATE,
    configure_logging,
)
from src.exceptions import (
    DataValidationError,
    FleetReliabilityException,
    ModelInferenceError,
    ModelNotFoundError,
    ThresholdCalibrationError,
)


def test_random_state_constant() -> None:
    """Verifies that RANDOM_STATE is an integer fixed to 42."""
    assert isinstance(RANDOM_STATE, int)
    assert RANDOM_STATE == 42


def test_logging_constants() -> None:
    """Verifies logger name, format string, and ISO 8601 date format constants."""
    assert LOGGER_NAME == "fleetreliability"
    assert (
        LOG_FORMAT
        == "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    assert LOG_DATE_FORMAT == "%Y-%m-%dT%H:%M:%S"


@pytest.mark.parametrize(
    "exc_cls",
    [
        DataValidationError,
        ModelNotFoundError,
        ModelInferenceError,
        ThresholdCalibrationError,
    ],
)
def test_exception_hierarchy(exc_cls: type[FleetReliabilityException]) -> None:
    """Verifies that every domain exception inherits from FleetReliabilityException."""
    assert issubclass(exc_cls, FleetReliabilityException)
    assert issubclass(exc_cls, Exception)

    message: str = f"Failure in {exc_cls.__name__}"
    with pytest.raises(FleetReliabilityException) as exc_info:
        raise exc_cls(message)

    assert isinstance(exc_info.value, exc_cls)
    assert str(exc_info.value) == message


def test_base_exception_empty_instantiation() -> None:
    """Verifies FleetReliabilityException supports empty initialization."""
    err: FleetReliabilityException = FleetReliabilityException()
    assert isinstance(err, Exception)
    assert str(err) == ""


def test_configure_logging_defaults_and_idempotency() -> None:
    """Verifies configure_logging sets name, level, formatter, and avoids duplicate handlers."""
    logger: logging.Logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()

    configured: logging.Logger = configure_logging()
    assert configured is logger
    assert configured.name == LOGGER_NAME
    assert configured.level == logging.INFO
    assert len(configured.handlers) == 1

    handler: logging.Handler = configured.handlers[0]
    assert handler.level == logging.INFO
    assert handler.formatter is not None
    assert handler.formatter._fmt == LOG_FORMAT
    assert handler.formatter.datefmt == LOG_DATE_FORMAT

    reconfigured: logging.Logger = configure_logging(level=logging.DEBUG)
    assert reconfigured is configured
    assert reconfigured.level == logging.DEBUG
    assert len(reconfigured.handlers) == 1
    assert reconfigured.handlers[0].level == logging.DEBUG


def test_configure_logging_formatted_output() -> None:
    """Verifies emitted log records follow ISO 8601 timestamp and structured format."""
    logger: logging.Logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()

    stream: io.StringIO = io.StringIO()
    stream_handler: logging.StreamHandler[io.StringIO] = logging.StreamHandler(stream)
    logger.addHandler(stream_handler)

    configure_logging(level=logging.WARNING)
    logger.info("ignored info message")
    logger.warning("pump cavitation detected")

    output: str = stream.getvalue().strip()
    assert "ignored info message" not in output
    pattern: str = (
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} \| WARNING  \| "
        r"fleetreliability:test_configure_logging_formatted_output:\d+ - "
        r"pump cavitation detected$"
    )
    assert re.match(pattern, output) is not None

    logger.handlers.clear()
