"""Unit tests for Monte Carlo cost simulation and dual-threshold optimization."""

import numpy as np
import pytest

from src.constants import RANDOM_STATE
from src.exceptions import DataValidationError, ThresholdCalibrationError
from src.montecarlo import (
    COST_DAMAGE_MEAN,
    COST_DAMAGE_STD,
    COST_FP_MEAN,
    COST_FP_STD,
    COST_TP_MEAN,
    COST_TP_STD,
    CRITICALITY_FACTORS,
    DOWNTIME_HOURLY_USD,
    MTTR_MEAN_HOURS,
    MTTR_STD_HOURS,
    RISK_AVERSION_LAMBDA,
    compute_bayes_threshold,
    evaluate_policy_cost,
    generate_cost_realizations,
    optimize_dual_thresholds_mc,
    run_monte_carlo_simulation,
)


def test_operational_constants_contract() -> None:
    """Verifies operational cost and reliability constants match FASE-02 §5."""
    assert COST_FP_MEAN == 350.0
    assert COST_FP_STD == 50.0
    assert COST_TP_MEAN == 1200.0
    assert COST_TP_STD == 200.0
    assert COST_DAMAGE_MEAN == 5000.0
    assert COST_DAMAGE_STD == 1000.0
    assert MTTR_MEAN_HOURS == 8.0
    assert MTTR_STD_HOURS == 2.0
    assert DOWNTIME_HOURLY_USD == 1500.0
    assert CRITICALITY_FACTORS == {"L": 1.0, "M": 1.5, "H": 2.5}
    assert RISK_AVERSION_LAMBDA == 0.15


def test_compute_bayes_threshold_analytical() -> None:
    """Verifies analytical Bayes threshold matches FASE-02 §5.3 derivation."""
    theta_bayes: float = compute_bayes_threshold(350.0, 1200.0, 23000.0)
    expected: float = 350.0 / (350.0 + (23000.0 - 1200.0))
    assert theta_bayes == pytest.approx(expected, rel=1e-12)
    assert theta_bayes == pytest.approx(0.0158013544, rel=1e-6)


def test_generate_cost_realizations_determinism_and_shapes() -> None:
    """Verifies RNG determinism with RANDOM_STATE=42 and output shapes."""
    eq_types: np.ndarray = np.array(["L", "M", "H", "M", "L"])
    run_a: dict[str, np.ndarray] = generate_cost_realizations(
        equipment_type=eq_types,
        n_iterations=50,
        random_seed=RANDOM_STATE,
    )
    run_b: dict[str, np.ndarray] = generate_cost_realizations(
        equipment_type=eq_types,
        n_iterations=50,
        random_seed=RANDOM_STATE,
    )

    expected_keys: set[str] = {"c_fp", "c_tp", "c_damage", "mttr", "c_fn"}
    assert set(run_a.keys()) == expected_keys
    for key in expected_keys:
        assert run_a[key].shape == (50, 5)
        assert np.array_equal(run_a[key], run_b[key])
        assert np.all(np.isfinite(run_a[key]))
        assert np.all(run_a[key] >= 0.0)


def test_generate_cost_realizations_analytical_lognormal_mttr() -> None:
    """Verifies MTTR follows analytical LogNormal(mean=8h, std=2h) moments."""
    eq_types: np.ndarray = np.array(["M"] * 50)
    realizations: dict[str, np.ndarray] = generate_cost_realizations(
        equipment_type=eq_types,
        n_iterations=4000,
        random_seed=RANDOM_STATE,
    )

    mttr: np.ndarray = realizations["mttr"]
    sigma_ln_expected: float = float(
        np.sqrt(np.log(1.0 + (MTTR_STD_HOURS**2 / MTTR_MEAN_HOURS**2)))
    )
    mu_ln_expected: float = float(
        np.log(MTTR_MEAN_HOURS) - 0.5 * (sigma_ln_expected**2)
    )

    log_mttr: np.ndarray = np.log(mttr)
    assert float(np.mean(log_mttr)) == pytest.approx(mu_ln_expected, rel=1e-2)
    assert float(np.std(log_mttr)) == pytest.approx(sigma_ln_expected, rel=1e-2)
    assert float(np.mean(mttr)) == pytest.approx(MTTR_MEAN_HOURS, rel=1e-2)
    assert float(np.std(mttr)) == pytest.approx(MTTR_STD_HOURS, rel=2e-2)


def test_generate_cost_realizations_criticality_scaling() -> None:
    """Verifies false negative costs scale strictly as H > M > L."""
    n_iter: int = 1000
    real_l: dict[str, np.ndarray] = generate_cost_realizations(
        equipment_type=np.array(["L"]),
        n_iterations=n_iter,
        random_seed=RANDOM_STATE,
    )
    real_m: dict[str, np.ndarray] = generate_cost_realizations(
        equipment_type=np.array(["M"]),
        n_iterations=n_iter,
        random_seed=RANDOM_STATE,
    )
    real_h: dict[str, np.ndarray] = generate_cost_realizations(
        equipment_type=np.array(["H"]),
        n_iterations=n_iter,
        random_seed=RANDOM_STATE,
    )

    # Identical RNG seed and shape guarantee identical c_damage and mttr draws
    assert np.array_equal(real_l["mttr"], real_m["mttr"])
    assert np.array_equal(real_m["mttr"], real_h["mttr"])
    assert np.all(real_h["c_fn"] > real_m["c_fn"])
    assert np.all(real_m["c_fn"] > real_l["c_fn"])

    assert float(np.mean(real_l["c_fn"])) == pytest.approx(
        COST_DAMAGE_MEAN + MTTR_MEAN_HOURS * DOWNTIME_HOURLY_USD * 1.0, rel=0.02
    )
    assert float(np.mean(real_m["c_fn"])) == pytest.approx(
        COST_DAMAGE_MEAN + MTTR_MEAN_HOURS * DOWNTIME_HOURLY_USD * 1.5, rel=0.02
    )
    assert float(np.mean(real_h["c_fn"])) == pytest.approx(
        COST_DAMAGE_MEAN + MTTR_MEAN_HOURS * DOWNTIME_HOURLY_USD * 2.5, rel=0.02
    )


def test_run_monte_carlo_simulation_contract_and_determinism() -> None:
    """Verifies 1-D simulation output structure, percentiles, and determinism."""
    y_true: np.ndarray = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    y_prob: np.ndarray = np.array(
        [0.02, 0.05, 0.12, 0.65, 0.85, 0.03, 0.08, 0.40, 0.01, 0.04]
    )
    eq_type: np.ndarray = np.array(["L", "M", "H", "M", "H", "L", "M", "H", "L", "M"])

    sim_a = run_monte_carlo_simulation(
        y_true=y_true,
        y_prob_calibrated=y_prob,
        equipment_type=eq_type,
        n_iterations=200,
        random_seed=RANDOM_STATE,
    )
    sim_b = run_monte_carlo_simulation(
        y_true=y_true,
        y_prob_calibrated=y_prob,
        equipment_type=eq_type,
        n_iterations=200,
        random_seed=RANDOM_STATE,
    )

    assert set(sim_a.keys()) == {
        "thresholds",
        "expected_cost",
        "cost_std",
        "p10_cost",
        "p90_cost",
        "p95_cost",
    }
    assert len(sim_a["thresholds"]) == 99
    assert np.array_equal(sim_a["thresholds"], sim_b["thresholds"])
    for key in ("expected_cost", "cost_std", "p10_cost", "p90_cost", "p95_cost"):
        assert len(sim_a[key]) == 99
        assert sim_a[key] == sim_b[key]

    assert np.all(np.array(sim_a["p10_cost"]) <= np.array(sim_a["p90_cost"]))
    assert np.all(np.array(sim_a["p90_cost"]) <= np.array(sim_a["p95_cost"]))


def test_optimize_dual_thresholds_mc_invariants_and_unconstrained_search() -> None:
    """Verifies strict theta_1 < theta_2, NaN masking, and unconstrained search."""
    # Construct separable data where optimal theta_1 lies well above [0.05, 0.08]
    y_true: np.ndarray = np.array([0] * 40 + [1] * 10)
    y_prob: np.ndarray = np.array([0.12] * 40 + [0.45] * 10)
    eq_type: np.ndarray = np.array(["M"] * 50)

    grid: np.ndarray = np.linspace(0.05, 0.95, 19)
    opt: dict[str, object] = optimize_dual_thresholds_mc(
        y_true=y_true,
        y_prob_calibrated=y_prob,
        equipment_type=eq_type,
        theta_1_grid=grid,
        theta_2_grid=grid,
        n_iterations=300,
        random_seed=RANDOM_STATE,
    )

    expected_keys: set[str] = {
        "theta_1_opt",
        "theta_2_opt",
        "objective_min",
        "expected_cost_opt",
        "p10_cost_opt",
        "p90_cost_opt",
        "p95_cost_opt",
        "theta_1_grid",
        "theta_2_grid",
        "expected_cost_surface",
        "p90_surface",
        "objective_surface",
    }
    assert set(opt.keys()) == expected_keys

    theta_1_opt: float = float(opt["theta_1_opt"])  # type: ignore[arg-type]
    theta_2_opt: float = float(opt["theta_2_opt"])  # type: ignore[arg-type]
    assert theta_1_opt < theta_2_opt
    # Confirm theta_1_opt is NOT artificially bounded to [0.05, 0.08]
    assert theta_1_opt > 0.12

    for scalar_key in (
        "theta_1_opt",
        "theta_2_opt",
        "objective_min",
        "expected_cost_opt",
        "p10_cost_opt",
        "p90_cost_opt",
        "p95_cost_opt",
    ):
        assert np.isfinite(float(opt[scalar_key]))  # type: ignore[arg-type]

    t1_grid: np.ndarray = opt["theta_1_grid"]  # type: ignore[assignment]
    t2_grid: np.ndarray = opt["theta_2_grid"]  # type: ignore[assignment]
    expected_surface: np.ndarray = opt["expected_cost_surface"]  # type: ignore[assignment]
    p90_surface: np.ndarray = opt["p90_surface"]  # type: ignore[assignment]
    obj_surface: np.ndarray = opt["objective_surface"]  # type: ignore[assignment]

    assert expected_surface.shape == (len(t1_grid), len(t2_grid))
    assert p90_surface.shape == (len(t1_grid), len(t2_grid))
    assert obj_surface.shape == (len(t1_grid), len(t2_grid))

    invalid_mask: np.ndarray = t1_grid[:, None] >= t2_grid[None, :]
    assert np.all(np.isnan(expected_surface[invalid_mask]))
    assert np.all(np.isnan(p90_surface[invalid_mask]))
    assert np.all(np.isnan(obj_surface[invalid_mask]))
    assert np.all(np.isfinite(obj_surface[~invalid_mask]))

    # Cross-check exact equivalence with evaluate_policy_cost at the optimum
    eval_opt: dict[str, float] = evaluate_policy_cost(
        y_true=y_true,
        y_prob_calibrated=y_prob,
        equipment_type=eq_type,
        theta_1=theta_1_opt,
        theta_2=theta_2_opt,
        n_iterations=300,
        random_seed=RANDOM_STATE,
    )
    assert eval_opt["expected_cost"] == pytest.approx(
        float(opt["expected_cost_opt"]), rel=1e-12  # type: ignore[arg-type]
    )
    assert eval_opt["p90_cost"] == pytest.approx(
        float(opt["p90_cost_opt"]), rel=1e-12  # type: ignore[arg-type]
    )
    assert eval_opt["p10_cost"] == pytest.approx(
        float(opt["p10_cost_opt"]), rel=1e-12  # type: ignore[arg-type]
    )
    assert eval_opt["p95_cost"] == pytest.approx(
        float(opt["p95_cost_opt"]), rel=1e-12  # type: ignore[arg-type]
    )
    assert eval_opt["objective_value"] == pytest.approx(
        float(opt["objective_min"]), rel=1e-12  # type: ignore[arg-type]
    )


def test_optimize_dual_thresholds_beats_static_and_reactive_policies() -> None:
    """Verifies dual-threshold policy lowers cost vs static and reactive baselines."""
    rng: np.random.Generator = np.random.default_rng(RANDOM_STATE)
    n_healthy: int = 180
    n_failing: int = 20
    y_true: np.ndarray = np.array([0] * n_healthy + [1] * n_failing)
    # Calibrated probabilities below static 0.50 threshold for failing units
    p_healthy: np.ndarray = rng.uniform(0.005, 0.04, size=n_healthy)
    p_failing: np.ndarray = rng.uniform(0.18, 0.42, size=n_failing)
    y_prob: np.ndarray = np.concatenate([p_healthy, p_failing])
    eq_type: np.ndarray = rng.choice(["L", "M", "H"], size=n_healthy + n_failing)

    opt = optimize_dual_thresholds_mc(
        y_true=y_true,
        y_prob_calibrated=y_prob,
        equipment_type=eq_type,
        theta_1_grid=np.linspace(0.02, 0.60, 30),
        theta_2_grid=np.linspace(0.02, 0.60, 30),
        n_iterations=500,
        random_seed=RANDOM_STATE,
    )

    static_policy = evaluate_policy_cost(
        y_true=y_true,
        y_prob_calibrated=y_prob,
        equipment_type=eq_type,
        theta_1=0.5,
        theta_2=0.5,
        n_iterations=500,
        random_seed=RANDOM_STATE,
    )
    reactive_policy = evaluate_policy_cost(
        y_true=y_true,
        y_prob_calibrated=y_prob,
        equipment_type=eq_type,
        theta_1=1.01,
        theta_2=1.02,
        n_iterations=500,
        random_seed=RANDOM_STATE,
    )

    assert opt["expected_cost_opt"] < static_policy["expected_cost"]
    assert opt["expected_cost_opt"] < reactive_policy["expected_cost"]
    assert opt["objective_min"] < static_policy["objective_value"]
    assert opt["objective_min"] < reactive_policy["objective_value"]


def test_evaluate_policy_cost_trizone_loss_matrix_exact() -> None:
    """Verifies exact tri-zone loss matrix assignment across all 6 cells."""
    # 6 samples covering all (zone, y_true) combinations under (0.2, 0.6)
    y_true: np.ndarray = np.array([0, 1, 0, 1, 0, 1])
    y_prob: np.ndarray = np.array([0.1, 0.1, 0.3, 0.3, 0.8, 0.8])
    eq_type: np.ndarray = np.array(["L", "M", "H", "L", "M", "H"])

    realizations = generate_cost_realizations(
        equipment_type=eq_type,
        n_iterations=100,
        random_seed=RANDOM_STATE,
    )
    expected_iter_cost: np.ndarray = (
        0.0
        + realizations["c_fn"][:, 1]
        + realizations["c_fp"][:, 2]
        + (realizations["c_fp"][:, 3] + realizations["c_tp"][:, 3])
        + realizations["c_tp"][:, 4]
        + realizations["c_tp"][:, 5]
    )

    metrics = evaluate_policy_cost(
        y_true=y_true,
        y_prob_calibrated=y_prob,
        equipment_type=eq_type,
        theta_1=0.2,
        theta_2=0.6,
        n_iterations=100,
        random_seed=RANDOM_STATE,
    )
    assert metrics["expected_cost"] == pytest.approx(
        float(np.mean(expected_iter_cost)), rel=1e-12
    )
    assert metrics["p90_cost"] == pytest.approx(
        float(np.percentile(expected_iter_cost, 90)), rel=1e-12
    )


@pytest.mark.parametrize(
    "bad_equipment",
    [
        np.array(["L", "X"]),
        np.array([]),
        np.array([["L", "M"]]),
        np.array(["l", "m"]),
        [None, "M"],
    ],
)
def test_generate_cost_realizations_invalid_equipment_raises(
    bad_equipment: object,
) -> None:
    """Verifies DataValidationError on unknown or malformed equipment_type."""
    with pytest.raises(DataValidationError):
        generate_cost_realizations(equipment_type=bad_equipment)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_iterations", [0, -10, True])
def test_generate_cost_realizations_invalid_iterations_raises(
    bad_iterations: int,
) -> None:
    """Verifies DataValidationError on non-positive or boolean n_iterations."""
    with pytest.raises(DataValidationError):
        generate_cost_realizations(
            equipment_type=np.array(["L", "M"]),
            n_iterations=bad_iterations,
        )


def test_simulation_and_policy_validation_errors() -> None:
    """Verifies DataValidationError and ThresholdCalibrationError across API."""
    with pytest.raises(DataValidationError):
        run_monte_carlo_simulation(
            y_true=np.array([0, 1]),
            y_prob_calibrated=np.array([0.2]),
            equipment_type=np.array(["L", "M"]),
        )

    with pytest.raises(DataValidationError):
        run_monte_carlo_simulation(
            y_true=np.array([0, 2]),
            y_prob_calibrated=np.array([0.2, 0.8]),
            equipment_type=np.array(["L", "M"]),
        )

    with pytest.raises(DataValidationError):
        run_monte_carlo_simulation(
            y_true=np.array([0, 1]),
            y_prob_calibrated=np.array([0.2, 1.2]),
            equipment_type=np.array(["L", "M"]),
        )

    with pytest.raises(DataValidationError):
        optimize_dual_thresholds_mc(
            y_true=np.array([0, 1]),
            y_prob_calibrated=np.array([0.2, 0.8]),
            equipment_type=np.array(["L", "M"]),
            risk_lambda=-0.1,
        )

    with pytest.raises(ThresholdCalibrationError):
        optimize_dual_thresholds_mc(
            y_true=np.array([0, 1]),
            y_prob_calibrated=np.array([0.2, 0.8]),
            equipment_type=np.array(["L", "M"]),
            theta_1_grid=np.array([0.5, 0.8]),
            theta_2_grid=np.array([0.2, 0.5]),
        )

    with pytest.raises(ThresholdCalibrationError):
        evaluate_policy_cost(
            y_true=np.array([0, 1]),
            y_prob_calibrated=np.array([0.2, 0.8]),
            equipment_type=np.array(["L", "M"]),
            theta_1=0.7,
            theta_2=0.3,
        )

    with pytest.raises(ThresholdCalibrationError):
        compute_bayes_threshold(
            cost_fp_mean=350.0, cost_tp_mean=2000.0, cost_fn_mean=1500.0
        )

    with pytest.raises(DataValidationError):
        compute_bayes_threshold(
            cost_fp_mean=-10.0, cost_tp_mean=1200.0, cost_fn_mean=23000.0
        )
