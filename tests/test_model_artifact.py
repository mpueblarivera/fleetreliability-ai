"""Contract and end-to-end inference tests for the serialized model artifact.

Validates the schema, calibration thresholds, feature contract, evaluation
metrics, operational cost metadata, and end-to-end inference behavior of
``models/best_model.bin`` produced in Phase 2.
"""

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from src.constants import RANDOM_STATE
from src.features import (
    PHYSICAL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    PhysicalFeatureTransformer,
    normalize_raw_dataframe,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODEL_ARTIFACT_PATH: Path = PROJECT_ROOT / "models" / "best_model.bin"

REQUIRED_ARTIFACT_KEYS: frozenset[str] = frozenset(
    {
        "pipeline",
        "model_version",
        "trained_at",
        "target_name",
        "thresholds",
        "features",
        "metrics",
        "metadata",
    }
)

REQUIRED_COST_MODEL_KEYS: frozenset[str] = frozenset(
    {
        "cost_fp_mean",
        "cost_tp_mean",
        "cost_damage_mean",
        "mttr_mean_hours",
        "downtime_hourly_usd",
        "criticality_factors",
    }
)

SNAKE_CASE_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]*$")


@pytest.fixture(scope="module")
def model_artifact() -> dict[str, Any]:
    """Loads the serialized model artifact once per test module.

    Returns:
        dict[str, Any]: Deserialized model bundle from ``models/best_model.bin``.
    """
    assert (
        MODEL_ARTIFACT_PATH.exists()
    ), f"Serialized model artifact not found at {MODEL_ARTIFACT_PATH}"
    loaded: Any = joblib.load(MODEL_ARTIFACT_PATH)
    assert isinstance(loaded, dict)
    return loaded


def test_model_artifact_exists_and_has_mandatory_keys(
    model_artifact: dict[str, Any],
) -> None:
    """Verifies artifact existence, non-empty size, and top-level schema keys.

    Args:
        model_artifact: Deserialized model artifact dictionary fixture.
    """
    assert MODEL_ARTIFACT_PATH.is_file()
    assert MODEL_ARTIFACT_PATH.stat().st_size > 0
    assert REQUIRED_ARTIFACT_KEYS.issubset(model_artifact.keys())

    pipeline: Any = model_artifact["pipeline"]
    assert isinstance(pipeline, CalibratedClassifierCV)
    assert hasattr(pipeline, "predict_proba")
    assert list(pipeline.classes_) == [0, 1]

    model_version: Any = model_artifact["model_version"]
    assert isinstance(model_version, str)
    assert len(model_version.strip()) > 0

    trained_at: Any = model_artifact["trained_at"]
    assert isinstance(trained_at, str)
    parsed_timestamp: datetime = datetime.fromisoformat(trained_at)
    assert parsed_timestamp.tzinfo is not None

    assert model_artifact["target_name"] == TARGET_COLUMN


def test_thresholds_are_valid_finite_ordered_floats(
    model_artifact: dict[str, Any],
) -> None:
    """Verifies dual thresholds satisfy ``0.0 < theta_warning < theta_critical < 1.0``.

    Args:
        model_artifact: Deserialized model artifact dictionary fixture.
    """
    thresholds: Any = model_artifact["thresholds"]
    assert isinstance(thresholds, dict)
    assert {"theta_warning", "theta_critical"}.issubset(thresholds.keys())

    theta_warning: Any = thresholds["theta_warning"]
    theta_critical: Any = thresholds["theta_critical"]

    assert isinstance(theta_warning, float)
    assert isinstance(theta_critical, float)
    assert math.isfinite(theta_warning)
    assert math.isfinite(theta_critical)
    assert 0.0 < theta_warning < theta_critical < 1.0


def test_features_match_canonical_snake_case_contract(
    model_artifact: dict[str, Any],
) -> None:
    """Verifies raw and derived feature lists match ``src.features`` exactly.

    Args:
        model_artifact: Deserialized model artifact dictionary fixture.
    """
    features: Any = model_artifact["features"]
    assert isinstance(features, dict)
    assert {"raw", "derived"}.issubset(features.keys())

    raw_features: Any = features["raw"]
    derived_features: Any = features["derived"]

    assert isinstance(raw_features, list)
    assert isinstance(derived_features, list)
    assert len(raw_features) == 6
    assert len(derived_features) == 4

    assert raw_features == RAW_FEATURE_COLUMNS
    assert derived_features == PHYSICAL_FEATURE_COLUMNS
    assert set(raw_features).isdisjoint(set(derived_features))

    for col in raw_features + derived_features:
        assert isinstance(col, str)
        assert SNAKE_CASE_PATTERN.match(col) is not None


def test_metrics_and_metadata_meet_quality_gates(
    model_artifact: dict[str, Any],
) -> None:
    """Verifies evaluation metrics meet Phase 2 quality gates and metadata is complete.

    Args:
        model_artifact: Deserialized model artifact dictionary fixture.
    """
    metrics: Any = model_artifact["metrics"]
    assert isinstance(metrics, dict)
    for required_metric in (
        "cv_pr_auc_mean",
        "holdout_pr_auc",
        "brier_score_calibrated",
    ):
        assert required_metric in metrics
        value: Any = metrics[required_metric]
        assert isinstance(value, (int, float))
        assert math.isfinite(float(value))

    assert float(metrics["cv_pr_auc_mean"]) > 0.70
    assert float(metrics["holdout_pr_auc"]) > 0.70
    assert 0.0 <= float(metrics["brier_score_calibrated"]) < 0.05

    metadata: Any = model_artifact["metadata"]
    assert isinstance(metadata, dict)
    assert metadata.get("random_state") == RANDOM_STATE
    assert isinstance(metadata.get("model_architecture"), str)
    assert len(metadata["model_architecture"].strip()) > 0

    bayes_threshold: Any = metadata.get("bayes_theoretical_threshold")
    assert isinstance(bayes_threshold, float)
    assert math.isfinite(bayes_threshold)
    assert 0.0 < bayes_threshold < 1.0

    cost_params: Any = metadata.get("cost_model_parameters")
    assert isinstance(cost_params, dict)
    assert REQUIRED_COST_MODEL_KEYS.issubset(cost_params.keys())

    for numeric_key in (
        "cost_fp_mean",
        "cost_tp_mean",
        "cost_damage_mean",
        "mttr_mean_hours",
        "downtime_hourly_usd",
    ):
        param_val: Any = cost_params[numeric_key]
        assert isinstance(param_val, (int, float))
        assert math.isfinite(float(param_val))
        assert float(param_val) > 0.0

    criticality_factors: Any = cost_params["criticality_factors"]
    assert isinstance(criticality_factors, dict)
    assert set(criticality_factors.keys()) == {"L", "M", "H"}
    assert (
        0.0
        < float(criticality_factors["L"])
        < float(criticality_factors["M"])
        < float(criticality_factors["H"])
    )


def test_end_to_end_inference_nominal_vs_extreme_failure(
    model_artifact: dict[str, Any],
) -> None:
    """Verifies end-to-end probability outputs for nominal vs extreme failure telemetry.

    Builds raw telemetry rows, applies ``PhysicalFeatureTransformer``, and asserts
    that ``predict_proba`` returns valid probabilities in ``[0.0, 1.0]`` with the
    extreme thermal/power/wear failure state strictly exceeding the healthy nominal state.

    Args:
        model_artifact: Deserialized model artifact dictionary fixture.
    """
    raw_inputs: pd.DataFrame = pd.DataFrame(
        [
            {
                "machine_type": "L",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551.0,
                "torque_nm": 42.8,
                "tool_wear_min": 20.0,
            },
            {
                "machine_type": "H",
                "air_temperature_k": 304.5,
                "process_temperature_k": 312.0,
                "rotational_speed_rpm": 1150.0,
                "torque_nm": 76.0,
                "tool_wear_min": 240.0,
            },
        ]
    )

    transformer: PhysicalFeatureTransformer = PhysicalFeatureTransformer()
    enriched_inputs: pd.DataFrame = transformer.fit_transform(raw_inputs)

    expected_columns: list[str] = (
        model_artifact["features"]["raw"] + model_artifact["features"]["derived"]
    )
    assert list(enriched_inputs.columns) == expected_columns

    pipeline: CalibratedClassifierCV = model_artifact["pipeline"]
    probas: np.ndarray = pipeline.predict_proba(enriched_inputs)

    assert probas.shape == (2, 2)
    assert np.all(np.isfinite(probas))
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-9)

    p_nominal: float = float(probas[0, 1])
    p_extreme_failure: float = float(probas[1, 1])

    assert 0.0 <= p_nominal <= 1.0
    assert 0.0 <= p_extreme_failure <= 1.0
    assert p_extreme_failure > p_nominal

    theta_critical: float = float(model_artifact["thresholds"]["theta_critical"])
    assert p_nominal < theta_critical
    assert p_extreme_failure >= theta_critical


def test_end_to_end_inference_from_raw_uci_headers(
    model_artifact: dict[str, Any],
) -> None:
    """Verifies raw UCI AI4I headers normalize cleanly into valid pipeline inference.

    Args:
        model_artifact: Deserialized model artifact dictionary fixture.
    """
    raw_uci_df: pd.DataFrame = pd.DataFrame(
        [
            {
                "Type": "M",
                "Air temperature [K]": 298.5,
                "Process temperature [K]": 309.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 50.0,
            },
            {
                "Type": "L",
                "Air temperature [K]": 303.0,
                "Process temperature [K]": 310.5,
                "Rotational speed [rpm]": 1180.0,
                "Torque [Nm]": 72.0,
                "Tool wear [min]": 235.0,
            },
        ]
    )

    normalized: pd.DataFrame = normalize_raw_dataframe(raw_uci_df)
    enriched: pd.DataFrame = PhysicalFeatureTransformer().fit_transform(normalized)

    pipeline: CalibratedClassifierCV = model_artifact["pipeline"]
    probas: np.ndarray = pipeline.predict_proba(enriched)

    assert probas.shape == (2, 2)
    assert np.all(np.isfinite(probas))
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-9)
    assert float(probas[1, 1]) > float(probas[0, 1])
