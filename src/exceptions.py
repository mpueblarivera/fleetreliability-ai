"""Domain-specific exception hierarchy for FleetReliability-AI.

Provides semantic exception classes so callers across ingestion, training,
calibration, and inference can handle domain failures explicitly.
"""


class FleetReliabilityException(Exception):
    """Base exception class for all FleetReliability-AI domain errors.

    Args:
        message: Optional human-readable error description.
        *args: Additional positional arguments forwarded to ``Exception``.

    Example:
        >>> err = FleetReliabilityException("Unexpected pipeline failure.")
        >>> str(err)
        'Unexpected pipeline failure.'
    """

    def __init__(self, message: str | None = None, *args: object) -> None:
        """Initializes the base domain exception.

        Args:
            message: Optional human-readable error description.
            *args: Additional positional arguments forwarded to ``Exception``.
        """
        if message is not None:
            super().__init__(message, *args)
        else:
            super().__init__(*args)


class DataValidationError(FleetReliabilityException):
    """Raised when input telemetry fails schema or physical range validation.

    Example:
        >>> raise DataValidationError("Torque must be non-negative.")
        Traceback (most recent call last):
            ...
        src.exceptions.DataValidationError: Torque must be non-negative.
    """


class ModelNotFoundError(FleetReliabilityException):
    """Raised when a serialized model artifact cannot be located on disk.

    Example:
        >>> raise ModelNotFoundError("Artifact missing at models/best_model.bin")
        Traceback (most recent call last):
            ...
        src.exceptions.ModelNotFoundError: Artifact missing at models/best_model.bin
    """


class ModelInferenceError(FleetReliabilityException):
    """Raised when model deserialization or prediction execution fails.

    Example:
        >>> raise ModelInferenceError("Corrupted model artifact binary.")
        Traceback (most recent call last):
            ...
        src.exceptions.ModelInferenceError: Corrupted model artifact binary.
    """


class ThresholdCalibrationError(FleetReliabilityException):
    """Raised when Monte Carlo cost optimization or threshold calibration fails.

    Example:
        >>> raise ThresholdCalibrationError("No feasible threshold pair found.")
        Traceback (most recent call last):
            ...
        src.exceptions.ThresholdCalibrationError: No feasible threshold pair found.
    """
