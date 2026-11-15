"""Unit tests for the Physical Feature Engineering module.

FleetReliability-AI.
"""

import numpy as np
import pandas as pd
import pytest

from src.constants import RANDOM_STATE
from src.features import (
    EPSILON,
    EXCLUDE_COLUMNS,
    FAILURE_MODES,
    PHYSICAL_FEATURE_COLUMNS,
    POWER_CONVERSION_FACTOR,
    RAW_COLUMN_MAPPING,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    PhysicalFeatureTransformer,
    compute_heat_stress_ratio,
    compute_mechanical_power,
    compute_strain_index,
    compute_temperature_difference,
    normalize_raw_dataframe,
    prepare_training_matrices,
)


@pytest.fixture
def synthetic_sample_df() -> pd.DataFrame:
    """Provides a synthetic raw AI4I 2020 DataFrame with analytically known outputs."""
    return pd.DataFrame(
        {
            "UDI": [1, 2],
            "Product ID": ["L47181", "M14860"],
            "Type": ["L", "M"],
            "Air temperature [K]": [300.0, 298.0],
            "Process temperature [K]": [310.0, 308.5],
            "Rotational speed [rpm]": [1500.0, 1200.0],
            "Torque [Nm]": [40.0, 50.0],
            "Tool wear [min]": [100.0, 200.0],
            "Machine failure": [0, 1],
            "TWF": [0, 0],
            "HDF": [0, 1],
            "PWF": [0, 0],
            "OSF": [0, 0],
            "RNF": [0, 0],
        }
    )


def test_module_constants_contract() -> None:
    """Verifies exported module-level physical constants and column definitions."""
    assert np.isclose(POWER_CONVERSION_FACTOR, (2.0 * np.pi) / 60_000.0)
    assert EPSILON == 1e-6
    assert TARGET_COLUMN == "machine_failure"
    assert FAILURE_MODES == ["TWF", "HDF", "PWF", "OSF", "RNF"]
    assert EXCLUDE_COLUMNS == ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    assert len(RAW_COLUMN_MAPPING) == 7
    assert len(RAW_FEATURE_COLUMNS) == 6
    assert len(PHYSICAL_FEATURE_COLUMNS) == 4


def test_normalize_raw_dataframe_maps_columns(
    synthetic_sample_df: pd.DataFrame,
) -> None:
    """Verifies that raw AI4I headers are normalized to canonical snake_case."""
    normalized = normalize_raw_dataframe(synthetic_sample_df)
    for col in RAW_FEATURE_COLUMNS + ["machine_failure"]:
        assert col in normalized.columns


def test_normalize_raw_dataframe_strips_utf8_bom_and_whitespace(
    synthetic_sample_df: pd.DataFrame,
) -> None:
    """Verifies UTF-8 BOM and surrounding whitespace are stripped from headers."""
    bom_df = synthetic_sample_df.rename(
        columns={"UDI": "\ufeffUDI", "Type": " Type ", "Torque [Nm]": " Torque [Nm]\t"}
    )
    normalized = normalize_raw_dataframe(bom_df)
    assert "UDI" in normalized.columns
    assert "\ufeffUDI" not in normalized.columns
    assert "machine_type" in normalized.columns
    assert "torque_nm" in normalized.columns


def test_pure_physics_functions_nominal_values() -> None:
    """Verifies that vectorized physics functions match analytical hand calculations."""
    # 1. Delta T: 310.0 - 300.0 = 10.0 K
    assert np.isclose(compute_temperature_difference(310.0, 300.0), 10.0, atol=1e-5)

    # 2. Mechanical Power kW: 40.0 * 1500 * (2*pi/60) / 1000 = 2*pi kW
    expected_power = (40.0 * 1500.0 * 2.0 * np.pi / 60.0) / 1000.0
    assert np.isclose(compute_mechanical_power(40.0, 1500.0), expected_power, atol=1e-5)

    # 3. Strain index: 100 * 40.0 = 4000.0 min*Nm
    assert np.isclose(compute_strain_index(100.0, 40.0), 4000.0, atol=1e-5)

    # 4. Heat stress ratio: (10.0 / 1500.0) * 1000.0 = 6.66667
    expected_ratio = (10.0 / 1500.0) * 1000.0
    assert np.isclose(
        compute_heat_stress_ratio(10.0, 1500.0), expected_ratio, atol=1e-5
    )


def test_compute_heat_stress_ratio_zero_rpm_edge_case() -> None:
    """Verifies that RPM=0 does not raise ZeroDivisionError thanks to EPSILON."""
    actual = compute_heat_stress_ratio(10.0, 0.0)
    assert not np.isnan(actual)
    assert not np.isinf(actual)
    assert actual > 0.0
    assert np.isclose(actual, (10.0 / EPSILON) * 1000.0)


def test_vectorized_functions_support_series_and_preserve_index() -> None:
    """Verifies all four physics functions accept Series and preserve custom indices."""
    custom_index = pd.Index([101, 205, 309], name="asset_row")
    torque = pd.Series([40.0, 50.0, 30.0], index=custom_index)
    rpm = pd.Series([1500.0, 0.0, 2000.0], index=custom_index)
    t_air = pd.Series([300.0, 298.0, 295.0], index=custom_index)
    t_proc = pd.Series([310.0, 308.5, 307.0], index=custom_index)
    wear = pd.Series([100.0, 200.0, 50.0], index=custom_index)

    power = compute_mechanical_power(torque, rpm)
    delta_t = compute_temperature_difference(t_proc, t_air)
    strain = compute_strain_index(wear, torque)
    heat_stress = compute_heat_stress_ratio(delta_t, rpm)

    for result in (power, delta_t, strain, heat_stress):
        assert isinstance(result, pd.Series)
        assert result.index.equals(custom_index)
        assert not result.isna().any()
        assert np.isfinite(result).all()

    assert np.isclose(power.iloc[0], 2.0 * np.pi, atol=1e-5)
    assert np.isclose(delta_t.iloc[1], 10.5, atol=1e-5)
    assert np.isclose(strain.iloc[2], 1500.0, atol=1e-5)
    assert np.isclose(heat_stress.iloc[1], (10.5 / EPSILON) * 1000.0)


def test_vectorized_functions_support_numpy_ndarray() -> None:
    """Verifies all four physics functions operate cleanly on NumPy ndarrays."""
    torque = np.array([40.0, 50.0])
    rpm = np.array([1500.0, 0.0])
    t_air = np.array([300.0, 298.0])
    t_proc = np.array([310.0, 308.5])
    wear = np.array([100.0, 200.0])

    power = compute_mechanical_power(torque, rpm)
    delta_t = compute_temperature_difference(t_proc, t_air)
    strain = compute_strain_index(wear, torque)
    heat_stress = compute_heat_stress_ratio(delta_t, rpm)

    for result in (power, delta_t, strain, heat_stress):
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert not np.isnan(result).any()
        assert np.isfinite(result).all()

    assert np.allclose(power, [2.0 * np.pi, 0.0], atol=1e-5)
    assert np.allclose(delta_t, [10.0, 10.5], atol=1e-5)
    assert np.allclose(strain, [4000.0, 10000.0], atol=1e-5)
    assert np.isclose(heat_stress[0], (10.0 / 1500.0) * 1000.0, atol=1e-5)
    assert np.isclose(heat_stress[1], (10.5 / EPSILON) * 1000.0)


def test_physical_feature_transformer_columns_and_shape(
    synthetic_sample_df: pd.DataFrame,
) -> None:
    """Verifies PhysicalFeatureTransformer outputs all derived features without NaNs."""
    df_norm = normalize_raw_dataframe(synthetic_sample_df)
    transformer = PhysicalFeatureTransformer()
    res = transformer.fit_transform(df_norm)

    assert len(res) == len(synthetic_sample_df)
    for col in PHYSICAL_FEATURE_COLUMNS:
        assert col in res.columns
        assert not res[col].isna().any()
        assert np.isfinite(res[col]).all()

    assert transformer.get_feature_names_out() == list(res.columns)
    assert transformer.get_feature_names_out(["machine_type", "torque_nm"]) == [
        "machine_type",
        "torque_nm",
        *PHYSICAL_FEATURE_COLUMNS,
    ]


def test_physical_feature_transformer_preserves_index(
    synthetic_sample_df: pd.DataFrame,
) -> None:
    """Verifies PhysicalFeatureTransformer preserves custom DataFrame indices."""
    df_norm = normalize_raw_dataframe(synthetic_sample_df)
    df_norm.index = pd.Index(["unit_alpha", "unit_beta"], name="asset_id")

    transformer = PhysicalFeatureTransformer()
    res = transformer.fit_transform(df_norm)

    assert res.index.equals(df_norm.index)


def test_input_dataframe_immutability(synthetic_sample_df: pd.DataFrame) -> None:
    """Verifies normalization, transformer, and split helper do not mutate inputs."""
    raw_snapshot = synthetic_sample_df.copy(deep=True)
    normalized = normalize_raw_dataframe(synthetic_sample_df)
    pd.testing.assert_frame_equal(synthetic_sample_df, raw_snapshot)

    norm_snapshot = normalized.copy(deep=True)
    transformer = PhysicalFeatureTransformer()
    _ = transformer.fit_transform(normalized)
    pd.testing.assert_frame_equal(normalized, norm_snapshot)

    df_extended = pd.concat([synthetic_sample_df] * 10, ignore_index=True)
    extended_snapshot = df_extended.copy(deep=True)
    _ = prepare_training_matrices(df_extended, test_size=0.2, random_state=RANDOM_STATE)
    pd.testing.assert_frame_equal(df_extended, extended_snapshot)


@pytest.mark.parametrize(
    "missing_col",
    [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ],
)
def test_physical_feature_transformer_missing_column_raises(
    synthetic_sample_df: pd.DataFrame,
    missing_col: str,
) -> None:
    """Verifies fit and transform raise ValueError when required columns are missing."""
    df_norm = normalize_raw_dataframe(synthetic_sample_df).drop(columns=[missing_col])
    transformer = PhysicalFeatureTransformer()

    with pytest.raises(ValueError, match=missing_col):
        transformer.fit(df_norm)

    with pytest.raises(ValueError, match=missing_col):
        transformer.transform(df_norm)


def test_prepare_training_matrices_missing_target_raises(
    synthetic_sample_df: pd.DataFrame,
) -> None:
    """Verifies prepare_training_matrices raises ValueError when target is missing."""
    df_no_target = synthetic_sample_df.drop(columns=["Machine failure"])
    with pytest.raises(ValueError, match="machine_failure"):
        prepare_training_matrices(df_no_target)


def test_prepare_training_matrices_missing_telemetry_raises(
    synthetic_sample_df: pd.DataFrame,
) -> None:
    """Verifies prepare_training_matrices raises ValueError when sensor column is missing."""
    df_no_torque = synthetic_sample_df.drop(columns=["Torque [Nm]"])
    with pytest.raises(ValueError, match="torque_nm"):
        prepare_training_matrices(df_no_torque)


def test_prepare_training_matrices_leakage_isolation(
    synthetic_sample_df: pd.DataFrame,
) -> None:
    """Verifies forbidden columns are excluded and 80/20 split contract holds."""
    # Replicate rows to allow minimum stratified 80/20 split in test
    df_extended = pd.concat([synthetic_sample_df] * 10, ignore_index=True)
    x_train_dev, x_test, y_train_dev, y_test = prepare_training_matrices(
        df_extended, test_size=0.2, random_state=42
    )

    forbidden_cols = [
        "UDI",
        "Product ID",
        "Machine failure",
        "machine_failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    for col in forbidden_cols:
        assert col not in x_train_dev.columns
        assert col not in x_test.columns

    # Verify presence of raw and derived physics features in X
    for col in RAW_FEATURE_COLUMNS + PHYSICAL_FEATURE_COLUMNS:
        assert col in x_train_dev.columns
        assert col in x_test.columns

    # Verify split proportions (80% Dev / 20% Test Holdout) and alignment
    assert len(x_train_dev) == 16
    assert len(x_test) == 4
    assert len(x_train_dev) == len(y_train_dev)
    assert len(x_test) == len(y_test)
    assert y_train_dev.mean() == pytest.approx(0.5)
    assert y_test.mean() == pytest.approx(0.5)
