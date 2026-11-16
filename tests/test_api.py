"""Integration tests for the FleetReliability-AI FastAPI REST service."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.features import (
    compute_heat_stress_ratio,
    compute_mechanical_power,
    compute_strain_index,
    compute_temperature_difference,
)
from src.predict import HealthResponse, PredictionResponse, app


def _base_payload(**overrides: Any) -> dict[str, Any]:
    """Builds a valid baseline telemetry payload with optional field overrides.

    Args:
        **overrides: Telemetry fields to override in the baseline payload.

    Returns:
        dict[str, Any]: Complete telemetry payload dictionary.
    """
    payload: dict[str, Any] = {
        "machine_id": "PUMP-NORMAL-01",
        "machine_type": "M",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551.0,
        "torque_nm": 42.8,
        "tool_wear_min": 10.0,
    }
    payload.update(overrides)
    return payload


def _assert_derived_physics(derived: dict[str, float], payload: dict[str, Any]) -> None:
    """Asserts that derived physics values match exact domain equations.

    Args:
        derived: The ``derived_physics`` dictionary from ``PredictionResponse``.
        payload: The original telemetry input payload.
    """
    expected_temp_diff: float = float(
        compute_temperature_difference(
            payload["process_temperature_k"],
            payload["air_temperature_k"],
        )
    )
    expected_power: float = float(
        compute_mechanical_power(
            payload["torque_nm"],
            payload["rotational_speed_rpm"],
        )
    )
    expected_strain: float = float(
        compute_strain_index(
            payload["tool_wear_min"],
            payload["torque_nm"],
        )
    )
    expected_heat_stress: float = float(
        compute_heat_stress_ratio(
            expected_temp_diff,
            payload["rotational_speed_rpm"],
        )
    )

    assert derived["mechanical_power_kw"] == pytest.approx(expected_power, rel=1e-6)
    assert derived["temperature_difference_k"] == pytest.approx(
        expected_temp_diff, rel=1e-6
    )
    assert derived["strain_index_min_nm"] == pytest.approx(expected_strain, rel=1e-6)
    assert derived["heat_stress_ratio"] == pytest.approx(expected_heat_stress, rel=1e-6)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provides an isolated TestClient running inside the FastAPI lifespan context.

    Yields:
        Iterator[TestClient]: Active FastAPI test client with loaded model artifact.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client: TestClient) -> None:
    """Verifies GET /health returns HTTP 200 OK with valid model metadata and thresholds."""
    response = client.get("/health")
    assert response.status_code == 200

    data: dict[str, Any] = response.json()
    validated: HealthResponse = HealthResponse.model_validate(data)

    assert validated.status == "HEALTHY"
    assert validated.model_loaded is True
    assert validated.uptime_seconds >= 0.0
    assert len(validated.model_version) > 0
    assert "theta_warning" in validated.thresholds
    assert "theta_critical" in validated.thresholds

    theta_warning: float = validated.thresholds["theta_warning"]
    theta_critical: float = validated.thresholds["theta_critical"]
    assert 0.0 <= theta_warning < theta_critical <= 1.0


def test_service_unavailable_when_model_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verifies GET /health and POST /predict return HTTP 503 when MODEL_PATH is missing."""
    missing_model_path: Path = tmp_path / "nonexistent_model.bin"
    monkeypatch.setenv("MODEL_PATH", str(missing_model_path))

    with TestClient(app) as unavailable_client:
        health_response = unavailable_client.get("/health")
        assert health_response.status_code == 503
        health_data: dict[str, Any] = health_response.json()
        validated_health: HealthResponse = HealthResponse.model_validate(health_data)
        assert validated_health.status == "UNHEALTHY"
        assert validated_health.model_loaded is False
        assert validated_health.thresholds == {}
        assert validated_health.uptime_seconds >= 0.0

        predict_response = unavailable_client.post("/predict", json=_base_payload())
        assert predict_response.status_code == 503
        assert "detail" in predict_response.json()


@pytest.mark.parametrize(
    ("payload", "expected_status", "expected_criticality"),
    [
        (
            _base_payload(),
            "NORMAL",
            "LOW",
        ),
        (
            _base_payload(
                machine_id="PUMP-INSPECT-02",
                machine_type="L",
                air_temperature_k=301.0,
                process_temperature_k=309.5,
                rotational_speed_rpm=1380.0,
                torque_nm=45.0,
                tool_wear_min=200.0,
            ),
            "INSPECTION_REQUIRED",
            "MEDIUM",
        ),
        (
            _base_payload(
                machine_id="MILL-SAG-CRITICAL",
                machine_type="H",
                air_temperature_k=304.5,
                process_temperature_k=313.8,
                rotational_speed_rpm=1200.0,
                torque_nm=95.0,
                tool_wear_min=240.0,
            ),
            "CRITICAL_INTERVENTION",
            "CRITICAL",
        ),
    ],
    ids=["normal_condition", "inspection_warning_condition", "critical_condition"],
)
def test_predict_operating_conditions(
    client: TestClient,
    payload: dict[str, Any],
    expected_status: str,
    expected_criticality: str,
) -> None:
    """Verifies POST /predict schema, exact derived physics, and threshold coherence."""
    health_data: dict[str, Any] = client.get("/health").json()
    theta_warning: float = health_data["thresholds"]["theta_warning"]
    theta_critical: float = health_data["thresholds"]["theta_critical"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data: dict[str, Any] = response.json()
    validated: PredictionResponse = PredictionResponse.model_validate(data)

    assert validated.machine_id == payload["machine_id"]
    assert validated.model_version == health_data["model_version"]
    assert 0.0 <= validated.failure_risk_score <= 1.0
    assert len(validated.recommended_cbm_action) > 0
    assert validated.operational_status == expected_status
    assert validated.criticality_level == expected_criticality

    if expected_status == "NORMAL":
        assert validated.failure_risk_score < theta_warning
    elif expected_status == "INSPECTION_REQUIRED":
        assert theta_warning <= validated.failure_risk_score < theta_critical
    else:
        assert validated.failure_risk_score >= theta_critical

    _assert_derived_physics(data["derived_physics"], payload)


def test_predict_validation_out_of_range_temperature(client: TestClient) -> None:
    """Verifies out-of-range air_temperature_k triggers HTTP 422."""
    payload: dict[str, Any] = _base_payload(
        machine_id="PUMP-FAIL-TEMP",
        machine_type="L",
        air_temperature_k=450.0,
    )
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors: list[dict[str, Any]] = response.json()["detail"]
    assert any(err["loc"][-1] == "air_temperature_k" for err in errors)


def test_predict_validation_invalid_machine_type(client: TestClient) -> None:
    """Verifies unsupported machine_type triggers HTTP 422."""
    payload: dict[str, Any] = _base_payload(
        machine_id="PUMP-FAIL-TYPE",
        machine_type="Z",
    )
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors: list[dict[str, Any]] = response.json()["detail"]
    assert any(err["loc"][-1] == "machine_type" for err in errors)


def test_predict_validation_thermal_incoherence(client: TestClient) -> None:
    """Verifies process_temperature_k < air_temperature_k - 5.0 triggers HTTP 422."""
    payload: dict[str, Any] = _base_payload(
        machine_id="PUMP-THERMAL-INCOHERENCE",
        air_temperature_k=310.0,
        process_temperature_k=300.0,
    )
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors: list[dict[str, Any]] = response.json()["detail"]
    assert any("Physical incoherence" in str(err.get("msg", "")) for err in errors)


def test_predict_validation_missing_required_fields(client: TestClient) -> None:
    """Verifies missing required telemetry fields trigger HTTP 422."""
    payload: dict[str, Any] = {
        "machine_id": "PUMP-MISSING-FIELDS",
        "process_temperature_k": 308.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    missing_fields: set[str] = {
        str(err["loc"][-1]) for err in response.json()["detail"]
    }
    assert {"machine_type", "air_temperature_k"}.issubset(missing_fields)


@pytest.mark.parametrize("endpoint", ["/docs", "/redoc"])
def test_openapi_documentation_endpoints(client: TestClient, endpoint: str) -> None:
    """Verifies Swagger UI (/docs) and ReDoc (/redoc) respond with HTTP 200 OK."""
    response = client.get(endpoint)
    assert response.status_code == 200
