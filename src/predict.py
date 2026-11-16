"""FastAPI Inference Microservice for FleetReliability-AI.

Exposes REST endpoints for health monitoring (``GET /health``) and real-time
predictive maintenance inference (``POST /predict``) backed by the calibrated
scikit-learn pipeline and tri-threshold operational CBM policy.
"""

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from src.constants import LOGGER_NAME, configure_logging
from src.features import RAW_FEATURE_COLUMNS, PhysicalFeatureTransformer

configure_logging()
logger: logging.Logger = logging.getLogger(f"{LOGGER_NAME}.api")

DEFAULT_MODEL_PATH: str = "models/best_model.bin"
DEFAULT_THRESHOLD_WARNING: float = 0.01
DEFAULT_THRESHOLD_CRITICAL: float = 0.76


class TelemetryPayload(BaseModel):
    """Sensor telemetry payload for machine condition assessment."""

    machine_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique identifier of the industrial asset or machine on site.",
        examples=["PUMP-SAG-04", "CRUSHER-01"],
    )
    machine_type: Literal["L", "M", "H"] = Field(
        ...,
        description=(
            "Equipment criticality / quality variant: L (Low), M (Medium), H (High)."
        ),
        examples=["M"],
    )
    air_temperature_k: float = Field(
        ...,
        ge=250.0,
        le=350.0,
        description=(
            "Ambient air temperature at site in Kelvin. Valid range: [250.0, 350.0] K."
        ),
        examples=[298.2],
    )
    process_temperature_k: float = Field(
        ...,
        ge=250.0,
        le=380.0,
        description=(
            "Internal process/oil temperature in Kelvin. "
            "Valid range: [250.0, 380.0] K."
        ),
        examples=[308.7],
    )
    rotational_speed_rpm: float = Field(
        ...,
        ge=500.0,
        le=3500.0,
        description=(
            "Main shaft angular rotational speed in RPM. "
            "Valid range: [500.0, 3500.0] RPM."
        ),
        examples=[1500.0],
    )
    torque_nm: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Applied shaft torque in Newton-meters. Valid range: [0.0, 200.0] Nm.",
        examples=[42.8],
    )
    tool_wear_min: float = Field(
        ...,
        ge=0.0,
        le=400.0,
        description=(
            "Cumulative tool wear time in minutes. Valid range: [0.0, 400.0] min."
        ),
        examples=[120.0],
    )

    @model_validator(mode="after")
    def validate_thermal_coherence(self) -> "TelemetryPayload":
        """Validates that process temperature is physically coherent with air temperature.

        Returns:
            TelemetryPayload: The validated telemetry payload instance.

        Raises:
            ValueError: If ``process_temperature_k`` is more than 5.0 K below
                ``air_temperature_k``.
        """
        if self.process_temperature_k < (self.air_temperature_k - 5.0):
            raise ValueError(
                "Physical incoherence: process_temperature_k "
                f"({self.process_temperature_k} K) cannot be significantly lower "
                f"than air_temperature_k ({self.air_temperature_k} K)."
            )
        return self


class DerivedPhysicsResponse(BaseModel):
    """Real-time engineered physical variables returned for diagnostic transparency."""

    mechanical_power_kw: float = Field(
        ..., description="Calculated mechanical shaft power in kilowatts (kW)."
    )
    temperature_difference_k: float = Field(
        ..., description="Thermal gradient between process and ambient air (K)."
    )
    strain_index_min_nm: float = Field(
        ...,
        description="Cumulative mechanical strain index (Tool Wear * Torque) in min*Nm.",
    )
    heat_stress_ratio: float = Field(
        ...,
        description="Specific heat stress ratio normalized by rotational speed (RPM).",
    )


class PredictionResponse(BaseModel):
    """Prescriptive and predictive Condition-Based Monitoring (CBM) response."""

    machine_id: str = Field(..., description="Evaluated asset identifier.")
    failure_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability of machine failure in [0.0, 1.0].",
    )
    operational_status: Literal[
        "NORMAL", "INSPECTION_REQUIRED", "CRITICAL_INTERVENTION"
    ] = Field(
        ...,
        description="Operational status determined by the optimized multi-threshold policy.",
    )
    recommended_cbm_action: str = Field(
        ...,
        description="Recommended Condition-Based Maintenance (CBM) action for operations.",
    )
    criticality_level: str = Field(
        ...,
        description="Composite event criticality level (LOW, MEDIUM, HIGH, CRITICAL).",
    )
    derived_physics: DerivedPhysicsResponse = Field(
        ...,
        description="Physical features computed during inference.",
    )
    model_version: str = Field(
        ..., description="Production model version that executed the prediction."
    )
    timestamp: datetime = Field(
        ..., description="ISO 8601 UTC timestamp of the inference event."
    )


class HealthResponse(BaseModel):
    """Operational health status of the inference microservice."""

    status: Literal["HEALTHY", "DEGRADED", "UNHEALTHY"] = Field(
        ..., description="Current service health state."
    )
    model_loaded: bool = Field(
        ..., description="Whether the inference pipeline artifact is loaded in memory."
    )
    model_version: str = Field(..., description="Loaded model artifact version.")
    thresholds: dict[str, float] = Field(
        ..., description="Active operational decision thresholds."
    )
    uptime_seconds: float = Field(
        ..., description="Elapsed time in seconds since service startup."
    )


def evaluate_operational_policy(
    prob: float,
    machine_type: str,
    threshold_warning: float = DEFAULT_THRESHOLD_WARNING,
    threshold_critical: float = DEFAULT_THRESHOLD_CRITICAL,
) -> tuple[Literal["NORMAL", "INSPECTION_REQUIRED", "CRITICAL_INTERVENTION"], str, str]:
    """Evaluates the tri-threshold policy and generates a prescriptive CBM recommendation.

    Args:
        prob: Calibrated failure probability in ``[0.0, 1.0]``.
        machine_type: Machine criticality tier (``"L"``, ``"M"``, or ``"H"``).
        threshold_warning: Lower threshold for preventive CBM inspection alerts.
            Defaults to ``0.01``.
        threshold_critical: Upper threshold for immediate derating and intervention.
            Defaults to ``0.76``.

    Returns:
        tuple[Literal["NORMAL", "INSPECTION_REQUIRED", "CRITICAL_INTERVENTION"], str, str]:
            Tuple of ``(operational_status, recommended_cbm_action, criticality_level)``.
    """
    operational_status: Literal[
        "NORMAL", "INSPECTION_REQUIRED", "CRITICAL_INTERVENTION"
    ]
    if prob < threshold_warning:
        operational_status = "NORMAL"
        action = (
            "Continue normal operation. Maintain standard shift lubrication "
            "and routine inspection schedule."
        )
        criticality = "LOW"
    elif threshold_warning <= prob < threshold_critical:
        operational_status = "INSPECTION_REQUIRED"
        action = (
            f"Preventive CBM alert for Class {machine_type} asset. Issue a "
            "non-destructive inspection work order for the next shift handover "
            "(bearing vibration analysis and component thermography)."
        )
        criticality = "MEDIUM" if machine_type == "L" else "HIGH"
    else:
        operational_status = "CRITICAL_INTERVENTION"
        action = (
            f"IMMINENT FAILURE RISK on critical Class {machine_type} asset. Execute "
            "immediate load derating (reduce RPM by 50%) and coordinate an immediate "
            "planned shutdown with the shift supervisor to prevent catastrophic damage."
        )
        criticality = "CRITICAL"

    return operational_status, action, criticality


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Manages application startup and shutdown lifecycle for model loading.

    Loads the serialized model artifact from ``MODEL_PATH`` (defaulting to
    ``models/best_model.bin``) into ``fastapi_app.state.model_artifact`` and
    records ``fastapi_app.state.startup_time``. If loading fails, sets
    ``model_artifact`` to ``None`` so health and prediction endpoints respond
    with HTTP 503 instead of crashing the process.

    Args:
        fastapi_app: Active FastAPI application instance.

    Yields:
        None: Control back to FastAPI while serving requests.
    """
    fastapi_app.state.startup_time = time.monotonic()
    fastapi_app.state.model_artifact = None

    model_path: str = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    logger.info(
        "Starting inference service. Loading model artifact from: %s", model_path
    )

    try:
        loaded_artifact: Any = joblib.load(model_path)
        if (
            not isinstance(loaded_artifact, dict)
            or "pipeline" not in loaded_artifact
            or "thresholds" not in loaded_artifact
        ):
            raise ValueError(
                "Invalid model artifact structure: expected dict with 'pipeline' "
                "and 'thresholds' keys."
            )
        fastapi_app.state.model_artifact = loaded_artifact
        logger.info(
            "Model artifact loaded successfully. Version: %s, Thresholds: %s",
            loaded_artifact.get("model_version"),
            loaded_artifact.get("thresholds"),
        )
    except (FileNotFoundError, OSError, EOFError, ValueError, KeyError) as exc:
        logger.critical("Failed to load model artifact from %s: %s", model_path, exc)
        fastapi_app.state.model_artifact = None

    yield

    logger.info(
        "Shutting down inference service. Releasing model artifact from memory."
    )
    fastapi_app.state.model_artifact = None
    fastapi_app.state.startup_time = None


app = FastAPI(
    title="FleetReliability-AI: Predictive Maintenance API",
    description="Condition-Based Monitoring (CBM) and failure risk inference web service.",
    version="0.1.0",
    lifespan=lifespan,
)


def _compute_uptime_seconds(fastapi_app: FastAPI) -> float:
    """Computes elapsed uptime in seconds from ``app.state.startup_time``."""
    startup_time: float | None = getattr(fastapi_app.state, "startup_time", None)
    if startup_time is None:
        return 0.0
    return max(0.0, time.monotonic() - startup_time)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and model readiness check",
)
async def health_check(request: Request) -> HealthResponse | JSONResponse:
    """Returns service health status, loaded model metadata, and uptime.

    Args:
        request: Incoming FastAPI request providing access to ``app.state``.

    Returns:
        HealthResponse | JSONResponse: HTTP 200 with ``status='HEALTHY'`` when
            the model artifact is loaded, or HTTP 503 with ``status='UNHEALTHY'``
            when the model artifact is unavailable.
    """
    artifact: dict[str, Any] | None = getattr(request.app.state, "model_artifact", None)
    uptime_seconds: float = round(_compute_uptime_seconds(request.app), 3)

    if artifact is None:
        unhealthy_payload = HealthResponse(
            status="UNHEALTHY",
            model_loaded=False,
            model_version="none",
            thresholds={},
            uptime_seconds=uptime_seconds,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=unhealthy_payload.model_dump(),
        )

    raw_thresholds: dict[str, Any] = artifact.get("thresholds", {})
    thresholds: dict[str, float] = {
        str(key): float(value) for key, value in raw_thresholds.items()
    }
    return HealthResponse(
        status="HEALTHY",
        model_loaded=True,
        model_version=str(artifact.get("model_version", "unknown")),
        thresholds=thresholds,
        uptime_seconds=uptime_seconds,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict machine failure risk and prescribe CBM action",
)
async def predict(payload: TelemetryPayload, request: Request) -> PredictionResponse:
    """Executes real-time failure risk inference and tri-threshold CBM policy.

    Args:
        payload: Validated machine telemetry payload.
        request: Incoming FastAPI request providing access to ``app.state``.

    Returns:
        PredictionResponse: Calibrated failure probability, operational status,
            prescriptive CBM recommendation, derived physics, and UTC timestamp.

    Raises:
        HTTPException: With status 503 if the model artifact is not loaded, or
            status 500 if an unexpected inference error occurs.
    """
    artifact: dict[str, Any] | None = getattr(request.app.state, "model_artifact", None)
    if artifact is None:
        logger.warning(
            "Prediction rejected for machine_id=%s: model artifact not loaded.",
            payload.machine_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Service unavailable.",
        )

    try:
        raw_row: dict[str, Any] = {
            "machine_type": payload.machine_type,
            "air_temperature_k": payload.air_temperature_k,
            "process_temperature_k": payload.process_temperature_k,
            "rotational_speed_rpm": payload.rotational_speed_rpm,
            "torque_nm": payload.torque_nm,
            "tool_wear_min": payload.tool_wear_min,
        }
        df_raw: pd.DataFrame = pd.DataFrame([raw_row])[RAW_FEATURE_COLUMNS]
        df_features: pd.DataFrame = PhysicalFeatureTransformer().transform(df_raw)

        row_features = df_features.iloc[0]
        derived_physics = DerivedPhysicsResponse(
            mechanical_power_kw=float(row_features["mechanical_power_kw"]),
            temperature_difference_k=float(row_features["temperature_difference_k"]),
            strain_index_min_nm=float(row_features["strain_index_min_nm"]),
            heat_stress_ratio=float(row_features["heat_stress_ratio"]),
        )

        pipeline = artifact["pipeline"]
        raw_prob: float = float(pipeline.predict_proba(df_features)[:, 1][0])
        failure_risk_score: float = min(max(raw_prob, 0.0), 1.0)

        thresholds: dict[str, Any] = artifact["thresholds"]
        theta_warning: float = float(
            thresholds.get("theta_warning", DEFAULT_THRESHOLD_WARNING)
        )
        theta_critical: float = float(
            thresholds.get("theta_critical", DEFAULT_THRESHOLD_CRITICAL)
        )

        operational_status, recommended_cbm_action, criticality_level = (
            evaluate_operational_policy(
                prob=failure_risk_score,
                machine_type=payload.machine_type,
                threshold_warning=theta_warning,
                threshold_critical=theta_critical,
            )
        )

        return PredictionResponse(
            machine_id=payload.machine_id,
            failure_risk_score=failure_risk_score,
            operational_status=operational_status,
            recommended_cbm_action=recommended_cbm_action,
            criticality_level=criticality_level,
            derived_physics=derived_physics,
            model_version=str(artifact.get("model_version", "unknown")),
            timestamp=datetime.now(timezone.utc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Unexpected inference error for machine_id=%s: %s",
            payload.machine_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference error while evaluating telemetry.",
        ) from exc
