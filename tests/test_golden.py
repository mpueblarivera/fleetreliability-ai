"""Golden dataset regression tests for the FastAPI inference service.

Validates end-to-end API contract compliance, probability calibration bounds,
tri-threshold operational CBM decision rules, and derived physical feature
calculations across canonical anchor cases in ``tests/golden_dataset.json``.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.predict import app

GOLDEN_DATASET_PATH: Path = Path(__file__).resolve().parent / "golden_dataset.json"

ZONE_ACTION_MARKERS: dict[str, str] = {
    "NORMAL": "normal operation",
    "INSPECTION_REQUIRED": "Preventive CBM alert",
    "CRITICAL_INTERVENTION": "IMMINENT FAILURE RISK",
}

PHYSICS_KEYS: tuple[str, ...] = (
    "mechanical_power_kw",
    "temperature_difference_k",
    "strain_index_min_nm",
    "heat_stress_ratio",
)


def _load_golden_cases() -> list[dict[str, Any]]:
    """Loads golden regression test cases from ``tests/golden_dataset.json``.

    Returns:
        list[dict[str, Any]]: List of golden anchor test case definitions.
    """
    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as fp:
        data: Any = json.load(fp)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            f"Expected non-empty JSON list in golden dataset at {GOLDEN_DATASET_PATH}."
        )
    return data


GOLDEN_CASES: list[dict[str, Any]] = _load_golden_cases()


@pytest.fixture(scope="module")
def api_client() -> Iterator[TestClient]:
    """Provides a FastAPI ``TestClient`` with lifespan startup/shutdown active.

    Yields:
        Iterator[TestClient]: Active test client with loaded model artifact.
    """
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    "golden_case",
    GOLDEN_CASES,
    ids=[str(case["case_id"]) for case in GOLDEN_CASES],
)
def test_golden_predict_contract_and_calibration(
    api_client: TestClient,
    golden_case: dict[str, Any],
) -> None:
    """Validates prediction response contract, risk bounds, CBM policy, and physics.

    Args:
        api_client: Lifespan-managed FastAPI ``TestClient`` fixture.
        golden_case: Parametrized golden dataset entry containing payload and
            expected operational, calibration, and physics assertions.
    """
    payload: dict[str, Any] = golden_case["payload"]
    expected_status: str = str(golden_case["expected_status"])
    expected_criticality: str = str(golden_case["expected_criticality"])
    risk_bounds: list[float] = [float(x) for x in golden_case["risk_bounds"]]
    expected_physics: dict[str, float] = {
        str(k): float(v) for k, v in golden_case["expected_physics"].items()
    }

    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200, (
        f"Case {golden_case['case_id']} returned HTTP {response.status_code}: "
        f"{response.text}"
    )

    body: dict[str, Any] = response.json()
    assert body["machine_id"] == payload["machine_id"]

    min_risk, max_risk = risk_bounds
    risk_score: float = float(body["failure_risk_score"])
    assert min_risk <= risk_score <= max_risk, (
        f"Case {golden_case['case_id']} failure_risk_score={risk_score} "
        f"outside calibrated bounds [{min_risk}, {max_risk}]"
    )

    assert body["operational_status"] == expected_status
    assert body["criticality_level"] == expected_criticality

    cbm_action: Any = body["recommended_cbm_action"]
    assert isinstance(cbm_action, str) and len(cbm_action.strip()) > 0
    expected_marker: str = ZONE_ACTION_MARKERS[expected_status]
    assert expected_marker in cbm_action, (
        f"Case {golden_case['case_id']} action '{cbm_action}' missing "
        f"expected zone marker '{expected_marker}'"
    )
    if expected_status in {"INSPECTION_REQUIRED", "CRITICAL_INTERVENTION"}:
        assert f"Class {payload['machine_type']}" in cbm_action

    derived_physics: dict[str, Any] = body["derived_physics"]
    for key in PHYSICS_KEYS:
        assert key in derived_physics
        assert float(derived_physics[key]) == pytest.approx(
            expected_physics[key], rel=1e-4
        )
