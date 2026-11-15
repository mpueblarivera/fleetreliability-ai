"""Domain-Informed Physical Feature Engineering Module for FleetReliability-AI.

Provides canonical column normalization, pure vectorized physics equations,
a scikit-learn compatible transformer, and a leakage-free stratified split
helper for predictive maintenance telemetry.
"""

import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.constants import LOGGER_NAME, RANDOM_STATE

logger: logging.Logger = logging.getLogger(f"{LOGGER_NAME}.features")

NumericInput = Union[float, np.ndarray, pd.Series]

# Physical constants
POWER_CONVERSION_FACTOR: float = (2.0 * np.pi) / (60.0 * 1000.0)
EPSILON: float = 1e-6

# Canonical raw column mapping from UCI AI4I 2020 headers to snake_case
RAW_COLUMN_MAPPING: dict[str, str] = {
    "Type": "machine_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}

RAW_FEATURE_COLUMNS: list[str] = [
    "machine_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

PHYSICAL_FEATURE_COLUMNS: list[str] = [
    "mechanical_power_kw",
    "temperature_difference_k",
    "strain_index_min_nm",
    "heat_stress_ratio",
]

REQUIRED_PHYSICS_COLUMNS: frozenset[str] = frozenset(
    {
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    }
)

TARGET_COLUMN: str = "machine_failure"

FAILURE_MODES: list[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]

EXCLUDE_COLUMNS: list[str] = ["UDI", "Product ID"] + FAILURE_MODES


def _validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: frozenset[str] | set[str],
    context: str,
) -> None:
    """Validates that an input object is a DataFrame containing required columns.

    Args:
        df: Candidate pandas DataFrame to validate.
        required_columns: Column names required for the operation.
        context: Descriptive operation label for log and error messages.

    Raises:
        ValueError: If `df` is not a pandas DataFrame or lacks required columns.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Expected pandas DataFrame for %s, got %s.", context, type(df))
        raise ValueError(f"Expected pandas DataFrame for {context}, got {type(df)}.")

    missing: list[str] = sorted(required_columns - set(df.columns))
    if missing:
        logger.error("Missing required columns for %s: %s", context, missing)
        raise ValueError(f"Missing required columns for {context}: {missing}")


def normalize_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes raw UCI AI4I 2020 column headers to canonical snake_case names.

    Strips leading/trailing whitespace and UTF-8 BOM characters (``\\ufeff``)
    from header names before applying ``RAW_COLUMN_MAPPING`` on a defensive copy.

    Args:
        df: Raw or partially normalized telemetry DataFrame.

    Returns:
        pd.DataFrame: Copy of `df` with standardized column names.

    Raises:
        ValueError: If `df` is not a pandas DataFrame.

    Example:
        >>> import pandas as pd
        >>> raw = pd.DataFrame({"\\ufeffUDI": [1], "Type": ["L"], "Torque [Nm]": [40.0]})
        >>> norm = normalize_raw_dataframe(raw)
        >>> list(norm.columns)
        ['UDI', 'machine_type', 'torque_nm']
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("normalize_raw_dataframe expected DataFrame, got %s.", type(df))
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}.")

    df_norm: pd.DataFrame = df.copy()
    df_norm.columns = [
        str(col).strip().lstrip("\ufeff").strip() for col in df_norm.columns
    ]
    normalized: pd.DataFrame = df_norm.rename(columns=RAW_COLUMN_MAPPING)
    logger.debug(
        "Normalized %d columns on telemetry DataFrame.", len(normalized.columns)
    )
    return normalized


def compute_mechanical_power(
    torque_nm: NumericInput,
    rotational_speed_rpm: NumericInput,
) -> NumericInput:
    """Computes mechanical shaft power in kilowatts (kW).

    Formula: ``P_mech = Torque [Nm] * Speed [rpm] * (2 * pi / 60000)``.

    Args:
        torque_nm: Applied shaft torque in Newton-meters.
        rotational_speed_rpm: Rotational speed in revolutions per minute.

    Returns:
        NumericInput: Mechanical power in kilowatts (kW).

    Example:
        >>> round(float(compute_mechanical_power(40.0, 1500.0)), 4)
        6.2832
    """
    return torque_nm * rotational_speed_rpm * POWER_CONVERSION_FACTOR


def compute_temperature_difference(
    process_temperature_k: NumericInput,
    air_temperature_k: NumericInput,
) -> NumericInput:
    """Computes thermal gradient between process and ambient air in Kelvin (K).

    Formula: ``Delta_T = Process_Temperature - Air_Temperature``.

    Args:
        process_temperature_k: Process temperature in Kelvin.
        air_temperature_k: Ambient air temperature in Kelvin.

    Returns:
        NumericInput: Temperature difference in Kelvin (K).

    Example:
        >>> compute_temperature_difference(310.0, 300.0)
        10.0
    """
    return process_temperature_k - air_temperature_k


def compute_strain_index(
    tool_wear_min: NumericInput,
    torque_nm: NumericInput,
) -> NumericInput:
    """Computes cumulative mechanical strain index in min*Nm.

    Formula: ``Strain_Index = Tool_Wear * Torque``.

    Args:
        tool_wear_min: Cumulative tool wear in minutes.
        torque_nm: Applied shaft torque in Newton-meters.

    Returns:
        NumericInput: Cumulative strain index in min*Nm.

    Example:
        >>> compute_strain_index(100.0, 40.0)
        4000.0
    """
    return tool_wear_min * torque_nm


def compute_heat_stress_ratio(
    temperature_difference_k: NumericInput,
    rotational_speed_rpm: NumericInput,
) -> NumericInput:
    """Computes specific heat stress ratio normalized by rotational speed.

    Formula: ``Heat_Stress_Ratio = (Delta_T / max(Rotational_Speed, EPSILON)) * 1000``.
    Clamps rotational speed from below at ``EPSILON`` to prevent division by zero.

    Args:
        temperature_difference_k: Thermal gradient in Kelvin.
        rotational_speed_rpm: Rotational speed in RPM.

    Returns:
        NumericInput: Scaled heat stress ratio.

    Example:
        >>> round(float(compute_heat_stress_ratio(10.0, 1500.0)), 4)
        6.6667
    """
    if isinstance(rotational_speed_rpm, (float, int, np.floating, np.integer)):
        denom: NumericInput = max(float(rotational_speed_rpm), EPSILON)
    elif isinstance(rotational_speed_rpm, pd.Series):
        denom = rotational_speed_rpm.clip(lower=EPSILON)
    else:
        denom = np.clip(rotational_speed_rpm, a_min=EPSILON, a_max=None)

    return (temperature_difference_k / denom) * 1000.0


class PhysicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that injects derived physical features.

    Expects normalized snake_case raw telemetry columns and appends the four
    derived thermodynamic and mechanical features without data leakage. Raises
    ``ValueError`` (aligned with scikit-learn estimator conventions) when
    required columns are missing.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "air_temperature_k": [300.0],
        ...     "process_temperature_k": [310.0],
        ...     "rotational_speed_rpm": [1500.0],
        ...     "torque_nm": [40.0],
        ...     "tool_wear_min": [100.0],
        ... })
        >>> transformer = PhysicalFeatureTransformer()
        >>> out = transformer.fit_transform(df)
        >>> "mechanical_power_kw" in out.columns
        True
    """

    def __init__(self) -> None:
        """Initializes the physical feature transformer."""
        self.feature_names_: list[str] = []

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "PhysicalFeatureTransformer":
        """Validates presence of required snake_case raw telemetry columns.

        Args:
            X: Input DataFrame with normalized raw telemetry columns.
            y: Unused target placeholder (scikit-learn estimator API compliance).

        Returns:
            PhysicalFeatureTransformer: Fitted transformer instance.

        Raises:
            ValueError: If `X` is not a DataFrame or any required raw telemetry
                column is missing.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     "air_temperature_k": [300.0],
            ...     "process_temperature_k": [310.0],
            ...     "rotational_speed_rpm": [1500.0],
            ...     "torque_nm": [40.0],
            ...     "tool_wear_min": [100.0],
            ... })
            >>> PhysicalFeatureTransformer().fit(df) is not None
            True
        """
        _validate_dataframe_columns(
            X, REQUIRED_PHYSICS_COLUMNS, context="physics feature transformation"
        )
        self.feature_names_ = list(X.columns) + [
            col for col in PHYSICAL_FEATURE_COLUMNS if col not in X.columns
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Computes and appends the four derived physical features.

        Args:
            X: Input DataFrame containing normalized raw telemetry columns.

        Returns:
            pd.DataFrame: Defensive copy of `X` enriched with derived physical
                features while preserving the original index.

        Raises:
            ValueError: If `X` is not a DataFrame or any required raw telemetry
                column is missing.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     "air_temperature_k": [300.0],
            ...     "process_temperature_k": [310.0],
            ...     "rotational_speed_rpm": [1500.0],
            ...     "torque_nm": [40.0],
            ...     "tool_wear_min": [100.0],
            ... })
            >>> transformer = PhysicalFeatureTransformer().fit(df)
            >>> list(transformer.transform(df).columns)[-4:]
            ['mechanical_power_kw', 'temperature_difference_k', 'strain_index_min_nm', 'heat_stress_ratio']
        """
        _validate_dataframe_columns(
            X, REQUIRED_PHYSICS_COLUMNS, context="physics feature transformation"
        )
        x_out: pd.DataFrame = X.copy()
        x_out["mechanical_power_kw"] = compute_mechanical_power(
            x_out["torque_nm"], x_out["rotational_speed_rpm"]
        )
        x_out["temperature_difference_k"] = compute_temperature_difference(
            x_out["process_temperature_k"], x_out["air_temperature_k"]
        )
        x_out["strain_index_min_nm"] = compute_strain_index(
            x_out["tool_wear_min"], x_out["torque_nm"]
        )
        x_out["heat_stress_ratio"] = compute_heat_stress_ratio(
            x_out["temperature_difference_k"], x_out["rotational_speed_rpm"]
        )
        self.feature_names_ = list(x_out.columns)
        logger.debug(
            "Computed %d physical features for %d rows.",
            len(PHYSICAL_FEATURE_COLUMNS),
            len(x_out),
        )
        return x_out

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Returns the list of output feature column names.

        Args:
            input_features: Optional list of input column names provided by
                scikit-learn pipelines.

        Returns:
            list[str]: Output feature column names in transformation order.

        Example:
            >>> transformer = PhysicalFeatureTransformer()
            >>> transformer.get_feature_names_out(["machine_type", "torque_nm"])
            ['machine_type', 'torque_nm', 'mechanical_power_kw', 'temperature_difference_k', 'strain_index_min_nm', 'heat_stress_ratio']
        """
        if input_features is not None:
            return list(input_features) + [
                col for col in PHYSICAL_FEATURE_COLUMNS if col not in input_features
            ]
        return list(self.feature_names_)


def prepare_training_matrices(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Normalizes columns, computes physical features, and performs an 80/20 stratified split.

    Drops identifier and post-mortem diagnostic columns (``UDI``, ``Product ID``,
    ``TWF``, ``HDF``, ``PWF``, ``OSF``, ``RNF``) to prevent target leakage, and
    returns a single stratified Dev / Test Holdout split.

    Args:
        df: Raw or normalized AI4I 2020 DataFrame.
        test_size: Fraction reserved for the locked test holdout set.
            Defaults to ``0.2``.
        random_state: Deterministic random seed for stratification.
            Defaults to ``RANDOM_STATE`` (``42``).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tuple
            ``(x_train_dev, x_test, y_train_dev, y_test)``.

    Raises:
        ValueError: If `df` is not a DataFrame, the ``machine_failure`` target
            column is missing, or any required telemetry column is missing.

    Example:
        >>> import pandas as pd
        >>> base = pd.DataFrame({
        ...     "UDI": list(range(1, 11)),
        ...     "Product ID": [f"L{i}" for i in range(10)],
        ...     "Type": ["L"] * 10,
        ...     "Air temperature [K]": [300.0] * 10,
        ...     "Process temperature [K]": [310.0] * 10,
        ...     "Rotational speed [rpm]": [1500.0] * 10,
        ...     "Torque [Nm]": [40.0] * 10,
        ...     "Tool wear [min]": [100.0] * 10,
        ...     "Machine failure": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ... })
        >>> x_dev, x_test, y_dev, y_test = prepare_training_matrices(base)
        >>> (len(x_dev), len(x_test))
        (8, 2)
    """
    df_norm: pd.DataFrame = normalize_raw_dataframe(df)
    _validate_dataframe_columns(
        df_norm, frozenset({TARGET_COLUMN}), context="training matrix preparation"
    )

    transformer = PhysicalFeatureTransformer()
    df_enriched: pd.DataFrame = transformer.fit_transform(df_norm)

    y: pd.Series = df_enriched[TARGET_COLUMN].copy()
    cols_to_drop: list[str] = [
        col for col in EXCLUDE_COLUMNS + [TARGET_COLUMN] if col in df_enriched.columns
    ]
    x: pd.DataFrame = df_enriched.drop(columns=cols_to_drop)

    x_train_dev, x_test, y_train_dev, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Prepared stratified split: train_dev=%d rows, test_holdout=%d rows.",
        len(x_train_dev),
        len(x_test),
    )
    return x_train_dev, x_test, y_train_dev, y_test
