"""Autonomous CLI training script for FleetReliability-AI.

Loads raw UCI AI4I 2020 telemetry data, performs leakage-free stratified
splitting and domain-informed physical feature engineering, evaluates
5-fold out-of-fold (OOF) cross-validation with isotonic probability
calibration, optimizes or validates dual CBM thresholds via Monte Carlo cost
simulation, evaluates once on the locked holdout test set, and serializes the
self-contained model artifact.
"""

import argparse
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import RANDOM_STATE, configure_logging
from src.exceptions import DataValidationError, ThresholdCalibrationError
from src.features import (
    PHYSICAL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_training_matrices,
)
from src.montecarlo import (
    COST_DAMAGE_MEAN,
    COST_FP_MEAN,
    COST_TP_MEAN,
    CRITICALITY_FACTORS,
    DOWNTIME_HOURLY_USD,
    MTTR_MEAN_HOURS,
    compute_bayes_threshold,
    optimize_dual_thresholds_mc,
)

MODEL_VERSION: str = "0.1.0"
AUTHOR_NAME: str = "Martín Puebla Rivera"
CV_FOLDS: int = 5

NUMERIC_FEATURE_COLUMNS: list[str] = [
    col for col in RAW_FEATURE_COLUMNS if col != "machine_type"
] + PHYSICAL_FEATURE_COLUMNS
CATEGORICAL_FEATURE_COLUMNS: list[str] = ["machine_type"]

logger: logging.Logger = configure_logging()


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser for autonomous model training.

    Returns:
        argparse.ArgumentParser: Configured parser with all training options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train, calibrate, and serialize the FleetReliability-AI predictive "
            "maintenance model."
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/ai4i2020.csv",
        help="Path to the raw UCI AI4I 2020 CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Target directory where the serialized model artifact will be saved.",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        default="best_model.bin",
        help="Filename for the serialized joblib model artifact.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of the dataset reserved for the locked test holdout split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Deterministic random seed for splitting, training, and simulation.",
    )
    parser.add_argument(
        "--theta-warning",
        type=float,
        default=None,
        help=(
            "Optional override for the warning threshold theta_1* "
            "(must be paired with --theta-critical)."
        ),
    )
    parser.add_argument(
        "--theta-critical",
        type=float,
        default=None,
        help=(
            "Optional override for the critical threshold theta_2* "
            "(must be paired with --theta-warning)."
        ),
    )
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable or disable 5-fold isotonic probability calibration "
            "(use --no-calibrate to disable)."
        ),
    )
    return parser


def _validate_test_size(test_size: float) -> float:
    """Validates that ``test_size`` is a finite float strictly within ``(0, 1)``."""
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not math.isfinite(float(test_size))
        or not (0.0 < float(test_size) < 1.0)
    ):
        raise DataValidationError(
            f"test_size must be strictly between 0.0 and 1.0, got {test_size!r}."
        )
    return float(test_size)


def _load_dataset(data_path: Path) -> pd.DataFrame:
    """Loads the raw telemetry CSV with ``utf-8-sig`` encoding and validates it.

    Args:
        data_path: Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded raw telemetry DataFrame.

    Raises:
        DataValidationError: If the file does not exist, cannot be read, or is empty.
    """
    if not data_path.exists() or not data_path.is_file():
        logger.error("Input dataset not found at %s.", data_path)
        raise DataValidationError(f"Input dataset not found at: {data_path}")

    try:
        df: pd.DataFrame = pd.read_csv(data_path, encoding="utf-8-sig")
    except Exception as err:
        logger.error("Failed to read CSV dataset from %s: %s", data_path, err)
        raise DataValidationError(
            f"Failed to read CSV dataset from {data_path}: {err}"
        ) from err

    if df.empty:
        logger.error("Input dataset at %s is empty.", data_path)
        raise DataValidationError(f"Input dataset at {data_path} is empty.")

    logger.info("Loaded dataset from %s with shape %s.", data_path, df.shape)
    return df


def build_base_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Constructs the Phase 2 champion preprocessing and classifier pipeline.

    Args:
        random_state: Deterministic random seed for the Random Forest classifier.

    Returns:
        Pipeline: Unfitted scikit-learn pipeline with ``ColumnTransformer`` and
            ``RandomForestClassifier``.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURE_COLUMNS),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=0.5,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_estimator(
    calibrate: bool = True,
    random_state: int = RANDOM_STATE,
) -> CalibratedClassifierCV | Pipeline:
    """Builds the trainable estimator, optionally wrapped in isotonic calibration.

    Args:
        calibrate: Whether to apply 5-fold isotonic probability calibration.
        random_state: Deterministic random seed for the underlying classifier.

    Returns:
        CalibratedClassifierCV | Pipeline: Estimator ready for cross-validation
            and final fitting.
    """
    base_pipeline: Pipeline = build_base_pipeline(random_state=random_state)
    if calibrate:
        return CalibratedClassifierCV(
            estimator=base_pipeline,
            method="isotonic",
            cv=CV_FOLDS,
        )
    return base_pipeline


def _validate_threshold_pair(
    theta_warning: float,
    theta_critical: float,
) -> tuple[float, float]:
    """Validates strict ordering ``0 < theta_warning < theta_critical < 1``."""
    for name, val in (
        ("theta_warning", theta_warning),
        ("theta_critical", theta_critical),
    ):
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float, np.number))
            or not math.isfinite(float(val))
        ):
            raise ThresholdCalibrationError(
                f"{name} must be a finite float, got {val!r}."
            )

    t_warn: float = float(theta_warning)
    t_crit: float = float(theta_critical)
    if not (0.0 < t_warn < t_crit < 1.0):
        raise ThresholdCalibrationError(
            "Thresholds must satisfy 0.0 < theta_warning < theta_critical < 1.0, "
            f"got theta_warning={t_warn}, theta_critical={t_crit}."
        )
    return t_warn, t_crit


def _resolve_thresholds(
    y_train_dev: pd.Series,
    oof_probs: np.ndarray,
    machine_types: np.ndarray,
    theta_warning: float | None,
    theta_critical: float | None,
    random_state: int,
) -> tuple[float, float]:
    """Resolves operational thresholds either from CLI overrides or Monte Carlo OOF.

    Args:
        y_train_dev: Ground-truth binary failure labels for the dev set.
        oof_probs: Out-of-fold predicted failure probabilities.
        machine_types: Equipment criticality labels (``"L"``, ``"M"``, ``"H"``).
        theta_warning: Optional CLI override for warning threshold.
        theta_critical: Optional CLI override for critical threshold.
        random_state: Deterministic random seed for Monte Carlo simulation.

    Returns:
        tuple[float, float]: Validated ``(theta_warning, theta_critical)`` pair.

    Raises:
        ThresholdCalibrationError: If only one CLI threshold is supplied or the
            resulting thresholds violate ``0.0 < theta_warning < theta_critical < 1.0``.
    """
    if (theta_warning is None) != (theta_critical is None):
        raise ThresholdCalibrationError(
            "Both --theta-warning and --theta-critical must be provided together, "
            "or omit both to optimize via Monte Carlo OOF."
        )

    if theta_warning is not None and theta_critical is not None:
        t_warn, t_crit = _validate_threshold_pair(theta_warning, theta_critical)
        logger.info(
            "Using CLI threshold overrides: theta_warning=%.4f, theta_critical=%.4f.",
            t_warn,
            t_crit,
        )
        return t_warn, t_crit

    logger.info(
        "Optimizing dual operational thresholds via Monte Carlo simulation on OOF probabilities."
    )
    mc_result: dict[str, Any] = optimize_dual_thresholds_mc(
        y_true=y_train_dev.to_numpy(),
        y_prob_calibrated=oof_probs,
        equipment_type=machine_types,
        random_seed=random_state,
    )
    t_warn, t_crit = _validate_threshold_pair(
        float(mc_result["theta_1_opt"]),
        float(mc_result["theta_2_opt"]),
    )
    logger.info(
        "Monte Carlo calibrated thresholds: theta_warning=%.4f, theta_critical=%.4f.",
        t_warn,
        t_crit,
    )
    return t_warn, t_crit


def train_and_serialize(
    data_path: Path,
    output_dir: Path,
    model_filename: str = "best_model.bin",
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    theta_warning: float | None = None,
    theta_critical: float | None = None,
    calibrate: bool = True,
) -> Path:
    """Executes the end-to-end leakage-free training, calibration, and serialization flow.

    Args:
        data_path: Path to the raw UCI AI4I 2020 CSV dataset.
        output_dir: Target directory for the serialized model artifact.
        model_filename: Output filename inside ``output_dir``.
        test_size: Holdout fraction reserved for final single evaluation.
        random_state: Deterministic random seed.
        theta_warning: Optional warning threshold override.
        theta_critical: Optional critical threshold override.
        calibrate: Whether to wrap the pipeline in 5-fold isotonic calibration.

    Returns:
        Path: Path to the serialized model artifact on disk.

    Raises:
        DataValidationError: If data loading, schema validation, or splitting fails.
        ThresholdCalibrationError: If threshold overrides or calibration are invalid.
    """
    validated_test_size: float = _validate_test_size(test_size)

    if (theta_warning is None) != (theta_critical is None):
        raise ThresholdCalibrationError(
            "Both --theta-warning and --theta-critical must be provided together, "
            "or omit both to optimize via Monte Carlo OOF."
        )
    if theta_warning is not None and theta_critical is not None:
        _validate_threshold_pair(theta_warning, theta_critical)

    df_raw: pd.DataFrame = _load_dataset(data_path)

    try:
        x_train_dev, x_test_holdout, y_train_dev, y_test_holdout = (
            prepare_training_matrices(
                df_raw,
                test_size=validated_test_size,
                random_state=random_state,
            )
        )
    except ValueError as err:
        logger.error("Failed to prepare training matrices: %s", err)
        raise DataValidationError(f"Invalid training dataset: {err}") from err

    if len(np.unique(y_train_dev)) < 2 or len(np.unique(y_test_holdout)) < 2:
        raise DataValidationError(
            "Both train_dev and test_holdout splits must contain classes 0 and 1."
        )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=random_state,
    )
    cv_estimator = build_estimator(calibrate=calibrate, random_state=random_state)

    logger.info(
        "Computing %d-fold OOF predictions on train_dev (%d rows, calibrate=%s).",
        CV_FOLDS,
        len(x_train_dev),
        calibrate,
    )
    oof_probs: np.ndarray = cross_val_predict(
        cv_estimator,
        x_train_dev,
        y_train_dev,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    t_warn, t_crit = _resolve_thresholds(
        y_train_dev=y_train_dev,
        oof_probs=oof_probs,
        machine_types=x_train_dev["machine_type"].to_numpy(),
        theta_warning=theta_warning,
        theta_critical=theta_critical,
        random_state=random_state,
    )

    fold_pr_aucs: list[float] = []
    fold_roc_aucs: list[float] = []
    fold_f2_scores: list[float] = []
    y_train_arr: np.ndarray = y_train_dev.to_numpy()

    for _, val_idx in cv.split(x_train_dev, y_train_arr):
        y_val: np.ndarray = y_train_arr[val_idx]
        p_val: np.ndarray = oof_probs[val_idx]
        fold_pr_aucs.append(float(average_precision_score(y_val, p_val)))
        fold_roc_aucs.append(float(roc_auc_score(y_val, p_val)))
        fold_f2_scores.append(
            float(
                fbeta_score(
                    y_val,
                    (p_val >= t_crit).astype(int),
                    beta=2.0,
                    zero_division=0,
                )
            )
        )

    cv_pr_auc_mean: float = float(np.mean(fold_pr_aucs))
    cv_pr_auc_std: float = float(np.std(fold_pr_aucs))
    cv_f2_score_mean: float = float(np.mean(fold_f2_scores))
    cv_roc_auc_mean: float = float(np.mean(fold_roc_aucs))
    brier_score_calibrated: float = float(brier_score_loss(y_train_arr, oof_probs))

    logger.info(
        "OOF CV metrics | cv_pr_auc_mean=%.4f (std=%.4f) | cv_roc_auc_mean=%.4f | "
        "cv_f2_score_mean=%.4f | brier_score_calibrated=%.6f",
        cv_pr_auc_mean,
        cv_pr_auc_std,
        cv_roc_auc_mean,
        cv_f2_score_mean,
        brier_score_calibrated,
    )

    final_model = build_estimator(calibrate=calibrate, random_state=random_state)
    logger.info(
        "Fitting final model on full train_dev split (%d rows).", len(x_train_dev)
    )
    final_model.fit(x_train_dev, y_train_dev)

    holdout_probs: np.ndarray = final_model.predict_proba(x_test_holdout)[:, 1]
    y_holdout_arr: np.ndarray = y_test_holdout.to_numpy()
    holdout_preds_critical: np.ndarray = (holdout_probs >= t_crit).astype(int)
    holdout_preds_warning: np.ndarray = (holdout_probs >= t_warn).astype(int)

    holdout_pr_auc: float = float(average_precision_score(y_holdout_arr, holdout_probs))
    holdout_roc_auc: float = float(roc_auc_score(y_holdout_arr, holdout_probs))
    holdout_f2_score: float = float(
        fbeta_score(y_holdout_arr, holdout_preds_critical, beta=2.0, zero_division=0)
    )
    holdout_f2_warning: float = float(
        fbeta_score(y_holdout_arr, holdout_preds_warning, beta=2.0, zero_division=0)
    )
    holdout_brier_score: float = float(brier_score_loss(y_holdout_arr, holdout_probs))
    cm: np.ndarray = confusion_matrix(
        y_holdout_arr, holdout_preds_critical, labels=[0, 1]
    )
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    logger.info(
        "Holdout evaluation (%d rows) | holdout_pr_auc=%.4f | holdout_roc_auc=%.4f | "
        "holdout_f2_score=%.4f | holdout_f2_warning=%.4f | holdout_brier_score=%.6f",
        len(y_holdout_arr),
        holdout_pr_auc,
        holdout_roc_auc,
        holdout_f2_score,
        holdout_f2_warning,
        holdout_brier_score,
    )
    logger.info(
        "Holdout confusion matrix at theta_critical=%.4f | TN=%d, FP=%d, FN=%d, TP=%d",
        t_crit,
        tn,
        fp,
        fn,
        tp,
    )

    model_architecture: str = (
        "RandomForestClassifier_Calibrated_Isotonic"
        if calibrate
        else "RandomForestClassifier_Uncalibrated"
    )
    artifact_payload: dict[str, Any] = {
        "pipeline": final_model,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target_name": TARGET_COLUMN,
        "thresholds": {
            "theta_warning": float(t_warn),
            "theta_critical": float(t_crit),
        },
        "features": {
            "raw": list(RAW_FEATURE_COLUMNS),
            "derived": list(PHYSICAL_FEATURE_COLUMNS),
        },
        "metrics": {
            "cv_pr_auc_mean": cv_pr_auc_mean,
            "cv_pr_auc_std": cv_pr_auc_std,
            "cv_f2_score_mean": cv_f2_score_mean,
            "cv_roc_auc_mean": cv_roc_auc_mean,
            "brier_score_calibrated": brier_score_calibrated,
            "holdout_pr_auc": holdout_pr_auc,
            "holdout_f2_score": holdout_f2_score,
            "holdout_f2_warning": holdout_f2_warning,
            "holdout_roc_auc": holdout_roc_auc,
            "holdout_brier_score": holdout_brier_score,
        },
        "metadata": {
            "model_architecture": model_architecture,
            "author": AUTHOR_NAME,
            "random_state": int(random_state),
            "bayes_theoretical_threshold": float(compute_bayes_threshold()),
            "cost_model_parameters": {
                "cost_fp_mean": float(COST_FP_MEAN),
                "cost_tp_mean": float(COST_TP_MEAN),
                "cost_damage_mean": float(COST_DAMAGE_MEAN),
                "mttr_mean_hours": float(MTTR_MEAN_HOURS),
                "downtime_hourly_usd": float(DOWNTIME_HOURLY_USD),
                "criticality_factors": dict(CRITICALITY_FACTORS),
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = output_dir / model_filename
    joblib.dump(artifact_payload, output_path, compress=3)
    artifact_size_bytes: int = output_path.stat().st_size
    logger.info(
        "Serialized model artifact saved to %s (%d bytes).",
        output_path,
        artifact_size_bytes,
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for training and serializing the CBM model.

    Args:
        argv: Optional list of CLI argument strings. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code (``0`` on success).
    """
    parser: argparse.ArgumentParser = build_argument_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    train_and_serialize(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        model_filename=args.model_filename,
        test_size=args.test_size,
        random_state=args.random_state,
        theta_warning=args.theta_warning,
        theta_critical=args.theta_critical,
        calibrate=args.calibrate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
