"""End-to-end and unit tests for the autonomous training CLI ``src.train``."""

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.constants import RANDOM_STATE
from src.exceptions import DataValidationError, ThresholdCalibrationError
from src.features import (
    PHYSICAL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    PhysicalFeatureTransformer,
)
from src.train import main

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

REQUIRED_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "cv_pr_auc_mean",
        "cv_pr_auc_std",
        "cv_f2_score_mean",
        "cv_roc_auc_mean",
        "brier_score_calibrated",
        "holdout_pr_auc",
        "holdout_f2_score",
        "holdout_f2_warning",
        "holdout_roc_auc",
        "holdout_brier_score",
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


def _create_synthetic_ai4i_csv(csv_path: Path, n_rows: int = 150) -> Path:
    """Creates a synthetic UCI AI4I 2020 CSV dataset with both classes and L/M/H types."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_failures: int = 30
    n_nominal: int = n_rows - n_failures

    types_nominal = rng.choice(["L", "M", "H"], size=n_nominal, p=[0.5, 0.3, 0.2])
    types_failure = rng.choice(["L", "M", "H"], size=n_failures, p=[0.4, 0.3, 0.3])
    machine_types = np.concatenate([types_nominal, types_failure])

    air_temp = np.concatenate(
        [
            rng.normal(298.5, 1.0, size=n_nominal),
            rng.normal(303.5, 1.0, size=n_failures),
        ]
    )
    process_temp = air_temp + np.concatenate(
        [
            rng.normal(10.5, 0.5, size=n_nominal),
            rng.normal(7.0, 0.6, size=n_failures),
        ]
    )
    speed = np.concatenate(
        [
            rng.normal(1520.0, 60.0, size=n_nominal),
            rng.normal(1180.0, 70.0, size=n_failures),
        ]
    )
    torque = np.concatenate(
        [
            rng.normal(40.0, 5.0, size=n_nominal),
            rng.normal(70.0, 6.0, size=n_failures),
        ]
    )
    tool_wear = np.concatenate(
        [
            rng.uniform(0.0, 120.0, size=n_nominal),
            rng.uniform(190.0, 250.0, size=n_failures),
        ]
    )
    failure = np.concatenate(
        [
            np.zeros(n_nominal, dtype=int),
            np.ones(n_failures, dtype=int),
        ]
    )

    df = pd.DataFrame(
        {
            "UDI": np.arange(1, n_rows + 1),
            "Product ID": [f"{t}{i:05d}" for i, t in enumerate(machine_types, start=1)],
            "Type": machine_types,
            "Air temperature [K]": np.round(air_temp, 2),
            "Process temperature [K]": np.round(process_temp, 2),
            "Rotational speed [rpm]": np.round(speed, 1),
            "Torque [Nm]": np.round(torque, 2),
            "Tool wear [min]": np.round(tool_wear, 1),
            "Machine failure": failure,
            "TWF": np.zeros(n_rows, dtype=int),
            "HDF": failure,
            "PWF": np.zeros(n_rows, dtype=int),
            "OSF": np.zeros(n_rows, dtype=int),
            "RNF": np.zeros(n_rows, dtype=int),
        }
    )
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


@pytest.fixture()
def synthetic_csv(tmp_path: Path) -> Path:
    """Writes a 150-row synthetic AI4I 2020 CSV inside ``tmp_path``."""
    return _create_synthetic_ai4i_csv(tmp_path / "ai4i_synth.csv", n_rows=150)


def test_train_cli_end_to_end_montecarlo_calibration(
    synthetic_csv: Path,
    tmp_path: Path,
) -> None:
    """Verifies end-to-end CLI execution, artifact schema, and Monte Carlo OOF thresholds."""
    output_dir: Path = tmp_path / "models_out"
    exit_code: int = main(
        [
            "--data-path",
            str(synthetic_csv),
            "--output-dir",
            str(output_dir),
            "--model-filename",
            "best_model.bin",
            "--test-size",
            "0.20",
            "--random-state",
            str(RANDOM_STATE),
        ]
    )
    assert exit_code == 0

    artifact_path: Path = output_dir / "best_model.bin"
    assert artifact_path.is_file()
    assert artifact_path.stat().st_size > 0

    artifact: dict[str, Any] = joblib.load(artifact_path)
    assert REQUIRED_ARTIFACT_KEYS.issubset(artifact.keys())

    assert isinstance(artifact["pipeline"], CalibratedClassifierCV)
    assert list(artifact["pipeline"].classes_) == [0, 1]
    assert isinstance(artifact["model_version"], str)
    assert len(artifact["model_version"].strip()) > 0

    parsed_ts = datetime.fromisoformat(artifact["trained_at"])
    assert parsed_ts.tzinfo is not None
    assert artifact["target_name"] == TARGET_COLUMN

    thresholds: dict[str, Any] = artifact["thresholds"]
    assert {"theta_warning", "theta_critical"}.issubset(thresholds.keys())
    assert isinstance(thresholds["theta_warning"], float)
    assert isinstance(thresholds["theta_critical"], float)
    assert 0.0 < thresholds["theta_warning"] < thresholds["theta_critical"] < 1.0

    features: dict[str, Any] = artifact["features"]
    assert features["raw"] == RAW_FEATURE_COLUMNS
    assert features["derived"] == PHYSICAL_FEATURE_COLUMNS

    metrics: dict[str, Any] = artifact["metrics"]
    assert REQUIRED_METRIC_KEYS.issubset(metrics.keys())
    for metric_name in REQUIRED_METRIC_KEYS:
        assert math.isfinite(float(metrics[metric_name]))

    metadata: dict[str, Any] = artifact["metadata"]
    assert (
        metadata["model_architecture"] == "RandomForestClassifier_Calibrated_Isotonic"
    )
    assert isinstance(metadata["author"], str) and len(metadata["author"]) > 0
    assert metadata["random_state"] == RANDOM_STATE
    assert 0.0 < float(metadata["bayes_theoretical_threshold"]) < 1.0
    cost_params: dict[str, Any] = metadata["cost_model_parameters"]
    assert REQUIRED_COST_MODEL_KEYS.issubset(cost_params.keys())

    # Verify serialized pipeline runs inference cleanly on enriched telemetry
    sample = pd.DataFrame(
        [
            {
                "machine_type": "L",
                "air_temperature_k": 298.5,
                "process_temperature_k": 309.0,
                "rotational_speed_rpm": 1520.0,
                "torque_nm": 40.0,
                "tool_wear_min": 30.0,
            }
        ]
    )
    enriched = PhysicalFeatureTransformer().fit_transform(sample)
    probas = artifact["pipeline"].predict_proba(enriched)
    assert probas.shape == (1, 2)
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-9)


def test_train_cli_explicit_threshold_overrides(
    synthetic_csv: Path,
    tmp_path: Path,
) -> None:
    """Verifies explicit ``--theta-warning`` and ``--theta-critical`` CLI overrides."""
    output_dir: Path = tmp_path / "models_override"
    exit_code: int = main(
        [
            "--data-path",
            str(synthetic_csv),
            "--output-dir",
            str(output_dir),
            "--theta-warning",
            "0.05",
            "--theta-critical",
            "0.65",
        ]
    )
    assert exit_code == 0

    artifact: dict[str, Any] = joblib.load(output_dir / "best_model.bin")
    assert artifact["thresholds"]["theta_warning"] == pytest.approx(0.05)
    assert artifact["thresholds"]["theta_critical"] == pytest.approx(0.65)


def test_train_cli_no_calibrate_flag(
    synthetic_csv: Path,
    tmp_path: Path,
) -> None:
    """Verifies ``--no-calibrate`` produces an uncalibrated scikit-learn Pipeline."""
    output_dir: Path = tmp_path / "models_uncal"
    exit_code: int = main(
        [
            "--data-path",
            str(synthetic_csv),
            "--output-dir",
            str(output_dir),
            "--no-calibrate",
        ]
    )
    assert exit_code == 0

    artifact: dict[str, Any] = joblib.load(output_dir / "best_model.bin")
    assert isinstance(artifact["pipeline"], Pipeline)
    assert (
        artifact["metadata"]["model_architecture"]
        == "RandomForestClassifier_Uncalibrated"
    )


def test_train_cli_missing_csv_raises_data_validation_error(tmp_path: Path) -> None:
    """Verifies missing input CSV raises ``DataValidationError``."""
    with pytest.raises(DataValidationError, match="not found"):
        main(
            [
                "--data-path",
                str(tmp_path / "does_not_exist.csv"),
                "--output-dir",
                str(tmp_path / "models"),
            ]
        )


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--theta-warning", "0.70", "--theta-critical", "0.50"],
        ["--theta-warning", "0.50", "--theta-critical", "0.50"],
        ["--theta-warning", "0.0", "--theta-critical", "0.50"],
        ["--theta-warning", "0.20", "--theta-critical", "1.0"],
        ["--theta-warning", "0.20"],
        ["--theta-critical", "0.70"],
    ],
)
def test_train_cli_invalid_thresholds_raises_threshold_calibration_error(
    synthetic_csv: Path,
    tmp_path: Path,
    cli_args: list[str],
) -> None:
    """Verifies invalid or partial threshold overrides raise ``ThresholdCalibrationError``."""
    with pytest.raises(ThresholdCalibrationError):
        main(
            [
                "--data-path",
                str(synthetic_csv),
                "--output-dir",
                str(tmp_path / "models"),
                *cli_args,
            ]
        )
