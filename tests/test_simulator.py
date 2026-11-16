"""Unit and end-to-end tests for the Stochastic Fleet Simulator (``src.fleet_simulator``)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from src.fleet_simulator import (
    FleetDegradationSimulator,
    compute_economic_comparison,
    main,
    parse_args,
    run_simulation,
)
from src.predict import TelemetryPayload, evaluate_operational_policy


def test_simulator_determinism_with_seed_42() -> None:
    """Verifies identical trajectories and latent failures across repeated seed=42 runs."""
    sim_a = FleetDegradationSimulator(
        num_machines=5, shift_hours=12, num_shifts=2, interval_hours=1, seed=42
    )
    sim_b = FleetDegradationSimulator(
        num_machines=5, shift_hours=12, num_shifts=2, interval_hours=1, seed=42
    )

    traj_a = sim_a.generate_trajectories()
    traj_b = sim_b.generate_trajectories()

    assert len(traj_a) == 5 * 24
    assert traj_a == traj_b


def test_simulator_trajectories_validate_against_telemetry_payload() -> None:
    """Verifies 100% of generated telemetry payloads pass TelemetryPayload validation."""
    simulator = FleetDegradationSimulator(
        num_machines=6, shift_hours=12, num_shifts=3, interval_hours=1, seed=42
    )
    trajectories = simulator.generate_trajectories()

    seen_types: set[str] = set()
    for step in trajectories:
        validated = TelemetryPayload.model_validate(step.payload)
        seen_types.add(validated.machine_type)
        assert validated.process_temperature_k >= (validated.air_temperature_k - 5.0)
        assert 0.0 <= validated.tool_wear_min <= 400.0
        assert step.latent_failure in (0, 1)

    assert seen_types == {"L", "M", "H"}


def test_simulator_tool_wear_monotonicity() -> None:
    """Verifies tool_wear_min advances monotonically for every machine."""
    simulator = FleetDegradationSimulator(
        num_machines=5, shift_hours=12, num_shifts=2, interval_hours=1, seed=42
    )
    trajectories = simulator.generate_trajectories()

    wear_history: dict[str, list[float]] = {}
    for step in trajectories:
        machine_id = str(step.payload["machine_id"])
        wear_history.setdefault(machine_id, []).append(
            float(step.payload["tool_wear_min"])
        )

    assert len(wear_history) == 5
    for machine_id, wears in wear_history.items():
        assert len(wears) == 24
        for prev, curr in zip(wears, wears[1:]):
            assert (
                curr >= prev
            ), f"Non-monotonic tool wear on {machine_id}: {prev} -> {curr}"


def test_economic_comparison_cost_consistency() -> None:
    """Verifies consistency between the three policy costs and net savings."""
    y_true = [0, 0, 1, 0, 1, 0, 1, 0]
    y_prob = [0.02, 0.05, 0.25, 0.03, 0.85, 0.01, 0.40, 0.04]
    equipment_types = ["L", "M", "H", "L", "H", "M", "H", "L"]

    econ = compute_economic_comparison(
        y_true=y_true,
        y_prob=y_prob,
        equipment_types=equipment_types,
        theta_warning=0.12,
        theta_critical=0.45,
        seed=42,
    )

    calibrated = econ["calibrated_multi_threshold_usd"]
    static_05 = econ["static_threshold_05_usd"]
    reactive = econ["reactive_run_to_failure_usd"]

    assert calibrated >= 0.0
    assert static_05 >= 0.0
    assert reactive >= 0.0
    assert econ["savings_vs_reactive_usd"] == pytest.approx(
        round(reactive - calibrated, 2), abs=1e-6
    )
    assert econ["savings_vs_static_05_usd"] == pytest.approx(
        round(static_05 - calibrated, 2), abs=1e-6
    )
    assert calibrated < reactive
    assert calibrated < static_05


def _make_stub_handler(theta_warning: float = 0.12, theta_critical: float = 0.45):
    """Creates a minimal HTTP request handler stubbing /health and /predict."""

    class _StubAPIHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, status_code: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send_json(404, {"detail": "Not found"})
                return
            self._send_json(
                200,
                {
                    "status": "HEALTHY",
                    "model_loaded": True,
                    "model_version": "stub-0.1.0",
                    "thresholds": {
                        "theta_warning": theta_warning,
                        "theta_critical": theta_critical,
                    },
                    "uptime_seconds": 1.0,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self._send_json(404, {"detail": "Not found"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload_dict = json.loads(raw_body)

            # Enforce strict Pydantic TelemetryPayload validation on stub server
            validated = TelemetryPayload.model_validate(payload_dict)

            # Deterministic synthetic risk score rising with wear and torque
            wear_factor = validated.tool_wear_min / 400.0
            torque_factor = validated.torque_nm / 200.0
            risk_score = min(1.0, max(0.0, 0.75 * wear_factor + 0.35 * torque_factor))

            op_status, action, criticality = evaluate_operational_policy(
                prob=risk_score,
                machine_type=validated.machine_type,
                threshold_warning=theta_warning,
                threshold_critical=theta_critical,
            )
            self._send_json(
                200,
                {
                    "machine_id": validated.machine_id,
                    "failure_risk_score": risk_score,
                    "operational_status": op_status,
                    "recommended_cbm_action": action,
                    "criticality_level": criticality,
                    "derived_physics": {
                        "mechanical_power_kw": 6.5,
                        "temperature_difference_k": (
                            validated.process_temperature_k
                            - validated.air_temperature_k
                        ),
                        "strain_index_min_nm": (
                            validated.tool_wear_min * validated.torque_nm
                        ),
                        "heat_stress_ratio": 7.0,
                    },
                    "model_version": "stub-0.1.0",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
            )

    return _StubAPIHandler


@pytest.fixture()
def stub_api_url():
    """Starts a local ThreadingHTTPServer stubbing /health and /predict."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_stub_handler())
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_end_to_end_simulation_with_stub_http_server(
    stub_api_url: str, tmp_path: Path
) -> None:
    """Verifies end-to-end simulation against a local HTTP stub and tmp_path output."""
    report_file = tmp_path / "nested" / "shift_report.json"

    report = run_simulation(
        api_url=stub_api_url,
        num_machines=5,
        shift_hours=12,
        num_shifts=2,
        interval_hours=1,
        seed=42,
        report_path=report_file,
        timeout=5.0,
    )

    assert report_file.exists()
    persisted = json.loads(report_file.read_text(encoding="utf-8"))
    assert persisted["simulation_config"]["total_events"] == 5 * 24
    assert persisted["simulation_config"]["thresholds"] == {
        "theta_warning": 0.12,
        "theta_critical": 0.45,
    }

    assert set(persisted["shift_summary"].keys()) == {"shift_1", "shift_2"}
    total_from_shifts = sum(
        sum(counts.values()) for counts in persisted["shift_summary"].values()
    )
    assert total_from_shifts == 5 * 24

    totals = persisted["total_status_counts"]
    assert sum(totals.values()) == 5 * 24
    expected_work_orders = (
        totals["INSPECTION_REQUIRED"] + totals["CRITICAL_INTERVENTION"]
    )
    assert len(persisted["work_orders"]) == expected_work_orders
    assert len(persisted["work_orders"]) > 0

    econ = persisted["economic_comparison"]
    for key in (
        "calibrated_multi_threshold_usd",
        "static_threshold_05_usd",
        "reactive_run_to_failure_usd",
        "savings_vs_reactive_usd",
        "savings_vs_static_05_usd",
    ):
        assert key in econ
    assert report["total_status_counts"] == totals


def test_cli_parse_args_and_main_end_to_end(stub_api_url: str, tmp_path: Path) -> None:
    """Verifies CLI argument parsing and main() execution using tmp_path."""
    report_file = tmp_path / "cli_shift_report.json"
    args = parse_args(
        [
            "--api-url",
            stub_api_url,
            "--num-machines",
            "3",
            "--shift-hours",
            "6",
            "--num-shifts",
            "2",
            "--interval-hours",
            "2",
            "--seed",
            "42",
            "--report-path",
            str(report_file),
            "--timeout",
            "3.0",
        ]
    )
    assert args.num_machines == 3
    assert args.interval_hours == 2

    exit_code = main(
        [
            "--api-url",
            stub_api_url,
            "--num-machines",
            "3",
            "--shift-hours",
            "6",
            "--num-shifts",
            "2",
            "--interval-hours",
            "2",
            "--seed",
            "42",
            "--report-path",
            str(report_file),
        ]
    )
    assert exit_code == 0
    assert report_file.exists()
