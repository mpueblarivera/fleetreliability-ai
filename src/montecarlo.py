"""Monte Carlo operational cost simulation and dual-threshold optimization.

FleetReliability-AI — Condition-Based Maintenance (CBM) decision engine.
Models stochastic maintenance costs and heavy-tailed LogNormal repair times
(MTTR) across asset criticality tiers ("L", "M", "H"), evaluates single- and
tri-zone operational policies, and optimizes the joint dual-threshold policy
``(theta_1*, theta_2*)`` minimizing expected cost plus tail risk (VaR P90).
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.constants import RANDOM_STATE, configure_logging
from src.exceptions import DataValidationError, ThresholdCalibrationError

logger: logging.Logger = configure_logging()

# Operational cost and reliability parameters (aligned with FASE-02 §5)
COST_FP_MEAN: float = 350.0
COST_FP_STD: float = 50.0
COST_TP_MEAN: float = 1200.0
COST_TP_STD: float = 200.0
COST_DAMAGE_MEAN: float = 5000.0
COST_DAMAGE_STD: float = 1000.0
MTTR_MEAN_HOURS: float = 8.0
MTTR_STD_HOURS: float = 2.0
DOWNTIME_HOURLY_USD: float = 1500.0
CRITICALITY_FACTORS: dict[str, float] = {"L": 1.0, "M": 1.5, "H": 2.5}
RISK_AVERSION_LAMBDA: float = 0.15


def _validate_n_iterations(n_iterations: int) -> int:
    """Validates that ``n_iterations`` is a positive integer."""
    if (
        isinstance(n_iterations, bool)
        or not isinstance(n_iterations, (int, np.integer))
        or int(n_iterations) < 1
    ):
        logger.error(
            "Invalid n_iterations=%r; expected positive integer.", n_iterations
        )
        raise DataValidationError(
            f"n_iterations must be a positive integer, got {n_iterations!r}."
        )
    return int(n_iterations)


def _validate_risk_lambda(risk_lambda: float) -> float:
    """Validates that ``risk_lambda`` is a finite non-negative float."""
    if (
        isinstance(risk_lambda, bool)
        or not isinstance(risk_lambda, (int, float, np.number))
        or not np.isfinite(float(risk_lambda))
        or float(risk_lambda) < 0.0
    ):
        logger.error("Invalid risk_lambda=%r; expected finite >= 0.", risk_lambda)
        raise DataValidationError(
            f"risk_lambda must be a finite non-negative number, got {risk_lambda!r}."
        )
    return float(risk_lambda)


def _validate_equipment_type(
    equipment_type: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validates equipment criticality tiers and maps them to ``K_Type`` factors."""
    eq_array: np.ndarray = np.asarray(equipment_type)
    if eq_array.ndim != 1 or eq_array.size == 0:
        logger.error(
            "Invalid equipment_type shape=%s; expected non-empty 1-D array.",
            eq_array.shape,
        )
        raise DataValidationError(
            f"equipment_type must be a non-empty 1-D array, got shape {eq_array.shape}."
        )

    valid_types: set[str] = set(CRITICALITY_FACTORS.keys())
    try:
        unique_values: set[object] = set(eq_array.tolist())
    except TypeError as err:
        logger.error("equipment_type contains unhashable elements: %s", err)
        raise DataValidationError(
            "equipment_type must contain only string tier labels 'L', 'M', or 'H'."
        ) from err

    if not unique_values.issubset(valid_types):
        invalid_values: list[str] = sorted(
            str(val) for val in unique_values - valid_types
        )
        logger.error(
            "Unsupported equipment_type values detected: %s",
            invalid_values,
        )
        raise DataValidationError(
            f"equipment_type contains unsupported values {invalid_values}. "
            f"Allowed values are {sorted(valid_types)}."
        )

    k_factors: np.ndarray = np.where(
        eq_array == "H",
        CRITICALITY_FACTORS["H"],
        np.where(
            eq_array == "M",
            CRITICALITY_FACTORS["M"],
            CRITICALITY_FACTORS["L"],
        ),
    ).astype(np.float64)
    return eq_array, k_factors


def _validate_simulation_inputs(
    y_true: ArrayLike,
    y_prob_calibrated: ArrayLike,
    equipment_type: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validates alignment and domain constraints for simulation input arrays."""
    eq_array, _ = _validate_equipment_type(equipment_type)

    y_true_raw: np.ndarray = np.asarray(y_true)
    y_prob_raw: np.ndarray = np.asarray(y_prob_calibrated)

    if y_true_raw.ndim != 1 or y_true_raw.size == 0:
        raise DataValidationError(
            f"y_true must be a non-empty 1-D array, got shape {y_true_raw.shape}."
        )
    if y_prob_raw.ndim != 1 or y_prob_raw.size == 0:
        raise DataValidationError(
            "y_prob_calibrated must be a non-empty 1-D array, "
            f"got shape {y_prob_raw.shape}."
        )
    if not (y_true_raw.size == y_prob_raw.size == eq_array.size):
        raise DataValidationError(
            "Input arrays must have identical lengths: "
            f"y_true={y_true_raw.size}, y_prob_calibrated={y_prob_raw.size}, "
            f"equipment_type={eq_array.size}."
        )

    try:
        y_true_float: np.ndarray = y_true_raw.astype(np.float64)
    except (TypeError, ValueError) as err:
        raise DataValidationError("y_true must contain binary values 0 or 1.") from err

    if not np.all(np.isfinite(y_true_float)) or not np.all(
        (y_true_float == 0.0) | (y_true_float == 1.0)
    ):
        raise DataValidationError(
            "y_true must contain only binary failure labels (0 or 1)."
        )

    if y_prob_raw.dtype == np.bool_:
        raise DataValidationError(
            "y_prob_calibrated must contain floating-point probabilities in [0.0, 1.0]."
        )
    try:
        y_prob_float: np.ndarray = y_prob_raw.astype(np.float64)
    except (TypeError, ValueError) as err:
        raise DataValidationError(
            "y_prob_calibrated must contain numeric probabilities in [0.0, 1.0]."
        ) from err

    if not np.all(np.isfinite(y_prob_float)) or np.any(
        (y_prob_float < 0.0) | (y_prob_float > 1.0)
    ):
        raise DataValidationError(
            "y_prob_calibrated must contain finite probabilities within [0.0, 1.0]."
        )

    return y_true_float.astype(np.int64), y_prob_float, eq_array


def _validate_threshold_grid(
    grid: ArrayLike | None,
    default_grid: np.ndarray,
    grid_name: str,
) -> np.ndarray:
    """Validates a candidate threshold grid or returns a copy of the default grid."""
    if grid is None:
        return default_grid.copy()

    try:
        grid_arr: np.ndarray = np.asarray(grid, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise DataValidationError(
            f"{grid_name} must be a 1-D array of probabilities in [0.0, 1.0]."
        ) from err

    if grid_arr.ndim != 1 or grid_arr.size == 0:
        raise DataValidationError(
            f"{grid_name} must be a non-empty 1-D array, got shape {grid_arr.shape}."
        )
    if not np.all(np.isfinite(grid_arr)) or np.any((grid_arr < 0.0) | (grid_arr > 1.0)):
        raise DataValidationError(
            f"{grid_name} values must be finite probabilities within [0.0, 1.0]."
        )
    return grid_arr


def _compute_lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """Computes analytical LogNormal ``(mu_ln, sigma_ln)`` from operational moments."""
    sigma_ln: float = float(np.sqrt(np.log(1.0 + (std**2 / mean**2))))
    mu_ln: float = float(np.log(mean) - 0.5 * (sigma_ln**2))
    return mu_ln, sigma_ln


def compute_bayes_threshold(
    cost_fp_mean: float = COST_FP_MEAN,
    cost_tp_mean: float = COST_TP_MEAN,
    cost_fn_mean: float = (
        COST_DAMAGE_MEAN
        + MTTR_MEAN_HOURS * DOWNTIME_HOURLY_USD * CRITICALITY_FACTORS["M"]
    ),
) -> float:
    """Computes the analytical Bayes-optimal decision threshold.

    Evaluates the theoretical minimum-expected-cost probability threshold under
    asymmetric maintenance costs:
    ``theta_bayes = C_FP / (C_FP + (C_FN - C_TP))``.

    Args:
        cost_fp_mean: Expected cost (USD) of a false positive inspection.
            Defaults to ``COST_FP_MEAN``.
        cost_tp_mean: Expected cost (USD) of a planned preventive intervention.
            Defaults to ``COST_TP_MEAN``.
        cost_fn_mean: Expected cost (USD) of an unplanned catastrophic failure.
            Defaults to the medium-criticality (``M``) analytical expectation
            ``$23,000``.

    Returns:
        float: Analytical Bayes threshold within ``[0.0, 1.0)``.

    Raises:
        DataValidationError: If any cost mean is non-finite or negative.
        ThresholdCalibrationError: If ``cost_fn_mean <= cost_tp_mean`` or the
            Bayes denominator is non-positive.

    Example:
        >>> round(compute_bayes_threshold(350.0, 1200.0, 23000.0), 4)
        0.0158
    """
    for name, value in (
        ("cost_fp_mean", cost_fp_mean),
        ("cost_tp_mean", cost_tp_mean),
        ("cost_fn_mean", cost_fn_mean),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise DataValidationError(
                f"{name} must be a finite non-negative number, got {value!r}."
            )

    c_fp: float = float(cost_fp_mean)
    c_tp: float = float(cost_tp_mean)
    c_fn: float = float(cost_fn_mean)

    if c_fn <= c_tp:
        raise ThresholdCalibrationError(
            "Bayes threshold requires cost_fn_mean > cost_tp_mean "
            f"(got cost_fn_mean={c_fn}, cost_tp_mean={c_tp})."
        )

    denominator: float = c_fp + (c_fn - c_tp)
    if denominator <= 0.0:
        raise ThresholdCalibrationError(
            f"Bayes threshold denominator must be positive, got {denominator}."
        )

    return float(c_fp / denominator)


def generate_cost_realizations(
    equipment_type: ArrayLike,
    n_iterations: int = 1000,
    random_seed: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Generates vectorized stochastic maintenance cost and MTTR realizations.

    Draws ``(n_iterations, n_samples)`` realizations using
    ``np.random.default_rng(random_seed)``. ``MTTR`` follows an analytical
    ``LogNormal(mu_ln, sigma_ln)`` parameterized from ``MTTR_MEAN_HOURS`` and
    ``MTTR_STD_HOURS``. Normal cost components (``c_fp``, ``c_tp``,
    ``c_damage``) use a non-negativity guard ``np.maximum(..., 0.0)`` solely to
    prevent extreme negative tail draws (5 to 7 standard deviations below the
    mean, probability < 3e-7) without distorting empirical moments.

    Args:
        equipment_type: 1-D array-like of asset criticality tiers (``"L"``,
            ``"M"``, or ``"H"``).
        n_iterations: Number of Monte Carlo stochastic realizations per sample.
            Defaults to ``1000``.
        random_seed: Seed for ``np.random.default_rng``. Defaults to
            ``RANDOM_STATE``.

    Returns:
        dict[str, np.ndarray]: Dictionary with ``(n_iterations, n_samples)``
            float64 arrays for keys ``"c_fp"``, ``"c_tp"``, ``"c_damage"``,
            ``"mttr"``, and ``"c_fn"``.

    Raises:
        DataValidationError: If ``equipment_type`` has invalid shape or unknown
            tier labels, or if ``n_iterations`` is not a positive integer.
    """
    n_iter: int = _validate_n_iterations(n_iterations)
    eq_array, k_factors = _validate_equipment_type(equipment_type)
    n_samples: int = int(eq_array.size)

    rng: np.random.Generator = np.random.default_rng(random_seed)
    mu_ln, sigma_ln = _compute_lognormal_params(MTTR_MEAN_HOURS, MTTR_STD_HOURS)

    shape: tuple[int, int] = (n_iter, n_samples)

    # Guard only against extreme (>5-sigma) negative normal tail without distorting moments.
    c_fp: np.ndarray = np.maximum(
        rng.normal(loc=COST_FP_MEAN, scale=COST_FP_STD, size=shape), 0.0
    )
    c_tp: np.ndarray = np.maximum(
        rng.normal(loc=COST_TP_MEAN, scale=COST_TP_STD, size=shape), 0.0
    )
    c_damage: np.ndarray = np.maximum(
        rng.normal(loc=COST_DAMAGE_MEAN, scale=COST_DAMAGE_STD, size=shape), 0.0
    )
    mttr: np.ndarray = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=shape)
    c_fn: np.ndarray = c_damage + (mttr * DOWNTIME_HOURLY_USD * k_factors[None, :])

    return {
        "c_fp": c_fp,
        "c_tp": c_tp,
        "c_damage": c_damage,
        "mttr": mttr,
        "c_fn": c_fn,
    }


def run_monte_carlo_simulation(
    y_true: ArrayLike,
    y_prob_calibrated: ArrayLike,
    equipment_type: ArrayLike,
    n_iterations: int = 1000,
    random_seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Runs a vectorized 1-D Monte Carlo simulation across decision thresholds.

    Evaluates binary decision thresholds ``np.linspace(0.01, 0.99, 99)`` under
    stochastic maintenance costs and LogNormal MTTR, quantifying expected
    operational cost and tail risk percentiles (``P10``, ``P90``, ``P95``).

    Args:
        y_true: Ground-truth binary failure labels (0 or 1).
        y_prob_calibrated: Calibrated failure probabilities in ``[0.0, 1.0]``.
        equipment_type: Machine criticality tiers (``"L"``, ``"M"``, or ``"H"``).
        n_iterations: Number of Monte Carlo realizations. Defaults to ``1000``.
        random_seed: Deterministic RNG seed. Defaults to ``RANDOM_STATE``.

    Returns:
        dict[str, Any]: Dictionary containing ``"thresholds"`` (``np.ndarray``)
            and lists of floats for ``"expected_cost"``, ``"cost_std"``,
            ``"p10_cost"``, ``"p90_cost"``, and ``"p95_cost"``.

    Raises:
        DataValidationError: If inputs fail shape, value, or tier validation.
    """
    y_true_arr, y_prob_arr, eq_array = _validate_simulation_inputs(
        y_true=y_true,
        y_prob_calibrated=y_prob_calibrated,
        equipment_type=equipment_type,
    )
    realizations: dict[str, np.ndarray] = generate_cost_realizations(
        equipment_type=eq_array,
        n_iterations=n_iterations,
        random_seed=random_seed,
    )

    thresholds: np.ndarray = np.linspace(0.01, 0.99, 99)
    pred_matrix: np.ndarray = y_prob_arr[None, :] >= thresholds[:, None]
    y_pos: np.ndarray = (y_true_arr == 1)[None, :]
    y_neg: np.ndarray = ~y_pos

    is_tp: np.ndarray = (pred_matrix & y_pos).astype(np.float64)
    is_fp: np.ndarray = (pred_matrix & y_neg).astype(np.float64)
    is_fn: np.ndarray = ((~pred_matrix) & y_pos).astype(np.float64)

    cost_matrix: np.ndarray = (
        (realizations["c_tp"] @ is_tp.T)
        + (realizations["c_fp"] @ is_fp.T)
        + (realizations["c_fn"] @ is_fn.T)
    )

    expected_cost: list[float] = np.mean(cost_matrix, axis=0).astype(float).tolist()
    cost_std: list[float] = np.std(cost_matrix, axis=0).astype(float).tolist()
    p10_cost: list[float] = (
        np.percentile(cost_matrix, 10, axis=0).astype(float).tolist()
    )
    p90_cost: list[float] = (
        np.percentile(cost_matrix, 90, axis=0).astype(float).tolist()
    )
    p95_cost: list[float] = (
        np.percentile(cost_matrix, 95, axis=0).astype(float).tolist()
    )

    logger.info(
        "Completed 1-D Monte Carlo simulation: n_samples=%d, n_iterations=%d, "
        "n_thresholds=%d, min_expected_cost=%.2f",
        y_true_arr.size,
        n_iterations,
        thresholds.size,
        min(expected_cost),
    )

    return {
        "thresholds": thresholds,
        "expected_cost": expected_cost,
        "cost_std": cost_std,
        "p10_cost": p10_cost,
        "p90_cost": p90_cost,
        "p95_cost": p95_cost,
    }


def _compute_trizone_cost_iterations(
    y_true_arr: np.ndarray,
    y_prob_arr: np.ndarray,
    realizations: dict[str, np.ndarray],
    theta_1: float,
    theta_2: float,
) -> np.ndarray:
    """Computes per-iteration fleet cost under the tri-zone loss matrix."""
    is_normal_fn: np.ndarray = (y_prob_arr < theta_1) & (y_true_arr == 1)
    is_insp_fp: np.ndarray = (
        (y_prob_arr >= theta_1) & (y_prob_arr < theta_2) & (y_true_arr == 0)
    )
    is_insp_tp: np.ndarray = (
        (y_prob_arr >= theta_1) & (y_prob_arr < theta_2) & (y_true_arr == 1)
    )
    is_crit: np.ndarray = y_prob_arr >= theta_2

    return (
        np.sum(realizations["c_fn"][:, is_normal_fn], axis=1)
        + np.sum(realizations["c_fp"][:, is_insp_fp], axis=1)
        + np.sum(
            realizations["c_fp"][:, is_insp_tp] + realizations["c_tp"][:, is_insp_tp],
            axis=1,
        )
        + np.sum(realizations["c_tp"][:, is_crit], axis=1)
    )


def optimize_dual_thresholds_mc(
    y_true: ArrayLike,
    y_prob_calibrated: ArrayLike,
    equipment_type: ArrayLike,
    theta_1_grid: ArrayLike | None = None,
    theta_2_grid: ArrayLike | None = None,
    risk_lambda: float = RISK_AVERSION_LAMBDA,
    n_iterations: int = 1000,
    random_seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Performs vectorized 2-D joint optimization of dual thresholds ``(theta_1, theta_2)``.

    Minimizes the risk-adjusted objective ``J(theta_1, theta_2) = E[Cost] +
    risk_lambda * P90_Cost`` over all feasible pairs satisfying ``theta_1 < theta_2``
    under the approved tri-zone loss matrix:

    - ``NORMAL`` (``p < theta_1``): ``0`` if ``y=0``, ``c_fn`` if ``y=1``.
    - ``INSPECTION_REQUIRED`` (``theta_1 <= p < theta_2``): ``c_fp`` if ``y=0``,
      ``c_fp + c_tp`` if ``y=1``.
    - ``CRITICAL_INTERVENTION`` (``p >= theta_2``): ``c_tp`` regardless of ``y``.

    Exploits the exact additive separability of the tri-zone loss matrix for
    ``theta_1 < theta_2`` into ``theta_1``-only and ``theta_2``-only terms,
    evaluating all feasible pairs without constructing a 4-D tensor.

    Args:
        y_true: Ground-truth binary failure labels (0 or 1).
        y_prob_calibrated: Calibrated failure probabilities in ``[0.0, 1.0]``.
        equipment_type: Machine criticality tiers (``"L"``, ``"M"``, or ``"H"``).
        theta_1_grid: Optional 1-D grid of candidate warning thresholds in
            ``[0.0, 1.0]``. Defaults to ``np.linspace(0.01, 0.99, 99)``.
        theta_2_grid: Optional 1-D grid of candidate critical thresholds in
            ``[0.0, 1.0]``. Defaults to ``np.linspace(0.01, 0.99, 99)``.
        risk_lambda: Risk aversion weight applied to ``P90_Cost``. Defaults to
            ``RISK_AVERSION_LAMBDA``.
        n_iterations: Number of Monte Carlo realizations. Defaults to ``1000``.
        random_seed: Deterministic RNG seed. Defaults to ``RANDOM_STATE``.

    Returns:
        dict[str, Any]: Optimization outputs including ``"theta_1_opt"``,
            ``"theta_2_opt"``, ``"objective_min"``, ``"expected_cost_opt"``,
            ``"p10_cost_opt"``, ``"p90_cost_opt"``, ``"p95_cost_opt"``,
            ``"theta_1_grid"``, ``"theta_2_grid"``,
            ``"expected_cost_surface"``, ``"p90_surface"``, and
            ``"objective_surface"``. Invalid surface cells where
            ``theta_1 >= theta_2`` contain ``np.nan``.

    Raises:
        DataValidationError: If arrays, grids, ``risk_lambda``, or
            ``n_iterations`` fail validation.
        ThresholdCalibrationError: If no grid pair satisfies
            ``theta_1 < theta_2`` or optimization yields non-finite surfaces.
    """
    y_true_arr, y_prob_arr, eq_array = _validate_simulation_inputs(
        y_true=y_true,
        y_prob_calibrated=y_prob_calibrated,
        equipment_type=equipment_type,
    )
    risk_lambda_val: float = _validate_risk_lambda(risk_lambda)
    default_grid: np.ndarray = np.linspace(0.01, 0.99, 99)
    t1_grid: np.ndarray = _validate_threshold_grid(
        theta_1_grid, default_grid, "theta_1_grid"
    )
    t2_grid: np.ndarray = _validate_threshold_grid(
        theta_2_grid, default_grid, "theta_2_grid"
    )

    valid_mask: np.ndarray = t1_grid[:, None] < t2_grid[None, :]
    if not np.any(valid_mask):
        logger.error("No feasible threshold pairs satisfy theta_1 < theta_2.")
        raise ThresholdCalibrationError(
            "No feasible threshold pairs satisfy strict inequality theta_1 < theta_2."
        )

    realizations: dict[str, np.ndarray] = generate_cost_realizations(
        equipment_type=eq_array,
        n_iterations=n_iterations,
        random_seed=random_seed,
    )

    c_fp: np.ndarray = realizations["c_fp"]
    c_tp: np.ndarray = realizations["c_tp"]
    c_fn: np.ndarray = realizations["c_fn"]

    y0: np.ndarray = y_true_arr == 0
    y1: np.ndarray = y_true_arr == 1

    # Additive decomposition for theta_1 < theta_2:
    # TotalCost(theta_1, theta_2) = A(theta_1) + B(theta_2)
    ge_t1: np.ndarray = (y_prob_arr[:, None] >= t1_grid[None, :]).astype(np.float64)
    ge_t2: np.ndarray = (y_prob_arr[:, None] >= t2_grid[None, :]).astype(np.float64)

    base_y1: np.ndarray = np.sum(c_fn[:, y1], axis=1, keepdims=True)
    w1_y0: np.ndarray = c_fp[:, y0]
    w1_y1: np.ndarray = c_fp[:, y1] + c_tp[:, y1] - c_fn[:, y1]
    a_mat: np.ndarray = base_y1 + (w1_y0 @ ge_t1[y0, :]) + (w1_y1 @ ge_t1[y1, :])

    w2_y0: np.ndarray = c_tp[:, y0] - c_fp[:, y0]
    w2_y1: np.ndarray = -c_fp[:, y1]
    b_mat: np.ndarray = (w2_y0 @ ge_t2[y0, :]) + (w2_y1 @ ge_t2[y1, :])

    n1: int = int(t1_grid.size)
    n2: int = int(t2_grid.size)
    expected_cost_surface: np.ndarray = np.full((n1, n2), np.nan, dtype=np.float64)
    p90_surface: np.ndarray = np.full((n1, n2), np.nan, dtype=np.float64)
    objective_surface: np.ndarray = np.full((n1, n2), np.nan, dtype=np.float64)

    for i in range(n1):
        valid_j: np.ndarray = valid_mask[i]
        if not np.any(valid_j):
            continue
        pair_costs: np.ndarray = a_mat[:, i : i + 1] + b_mat[:, valid_j]
        mean_row: np.ndarray = np.mean(pair_costs, axis=0)
        p90_row: np.ndarray = np.percentile(pair_costs, 90, axis=0)
        expected_cost_surface[i, valid_j] = mean_row
        p90_surface[i, valid_j] = p90_row
        objective_surface[i, valid_j] = mean_row + (risk_lambda_val * p90_row)

    if not np.any(np.isfinite(objective_surface)):
        raise ThresholdCalibrationError(
            "Dual-threshold optimization produced no finite objective values."
        )

    flat_opt: int = int(np.nanargmin(objective_surface))
    i_opt, j_opt = np.unravel_index(flat_opt, objective_surface.shape)

    theta_1_opt: float = float(t1_grid[i_opt])
    theta_2_opt: float = float(t2_grid[j_opt])

    opt_costs: np.ndarray = a_mat[:, i_opt] + b_mat[:, j_opt]
    expected_cost_opt: float = float(expected_cost_surface[i_opt, j_opt])
    p90_cost_opt: float = float(p90_surface[i_opt, j_opt])
    objective_min: float = float(objective_surface[i_opt, j_opt])
    p10_cost_opt: float = float(np.percentile(opt_costs, 10))
    p95_cost_opt: float = float(np.percentile(opt_costs, 95))

    logger.info(
        "Optimized dual thresholds: theta_1*=%.4f, theta_2*=%.4f, "
        "E[Cost]=%.2f, P90=%.2f, J_min=%.2f",
        theta_1_opt,
        theta_2_opt,
        expected_cost_opt,
        p90_cost_opt,
        objective_min,
    )

    return {
        "theta_1_opt": theta_1_opt,
        "theta_2_opt": theta_2_opt,
        "objective_min": objective_min,
        "expected_cost_opt": expected_cost_opt,
        "p10_cost_opt": p10_cost_opt,
        "p90_cost_opt": p90_cost_opt,
        "p95_cost_opt": p95_cost_opt,
        "theta_1_grid": t1_grid,
        "theta_2_grid": t2_grid,
        "expected_cost_surface": expected_cost_surface,
        "p90_surface": p90_surface,
        "objective_surface": objective_surface,
    }


def evaluate_policy_cost(
    y_true: ArrayLike,
    y_prob_calibrated: ArrayLike,
    equipment_type: ArrayLike,
    theta_1: float,
    theta_2: float,
    risk_lambda: float = RISK_AVERSION_LAMBDA,
    n_iterations: int = 1000,
    random_seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """Evaluates a fixed dual-threshold policy ``(theta_1, theta_2)`` under Monte Carlo uncertainty.

    Supports both strict tri-zone policies (``theta_1 < theta_2``) and degenerate
    benchmark policies such as the static binary baseline (``theta_1=0.5,
    theta_2=0.5``) and the run-to-failure reactive policy (``theta_1=1.01,
    theta_2=1.02``).

    Args:
        y_true: Ground-truth binary failure labels (0 or 1).
        y_prob_calibrated: Calibrated failure probabilities in ``[0.0, 1.0]``.
        equipment_type: Machine criticality tiers (``"L"``, ``"M"``, or ``"H"``).
        theta_1: Warning threshold separating ``NORMAL`` from
            ``INSPECTION_REQUIRED``.
        theta_2: Critical threshold separating ``INSPECTION_REQUIRED`` from
            ``CRITICAL_INTERVENTION``. Must satisfy ``theta_1 <= theta_2``.
        risk_lambda: Risk aversion weight applied to ``P90_Cost``. Defaults to
            ``RISK_AVERSION_LAMBDA``.
        n_iterations: Number of Monte Carlo realizations. Defaults to ``1000``.
        random_seed: Deterministic RNG seed. Defaults to ``RANDOM_STATE``.

    Returns:
        dict[str, float]: Policy metrics ``"expected_cost"``, ``"cost_std"``,
            ``"p10_cost"``, ``"p90_cost"``, ``"p95_cost"``, and
            ``"objective_value"``.

    Raises:
        DataValidationError: If arrays, thresholds, ``risk_lambda``, or
            ``n_iterations`` are invalid or non-finite.
        ThresholdCalibrationError: If ``theta_1 > theta_2``.
    """
    for name, val in (("theta_1", theta_1), ("theta_2", theta_2)):
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float, np.number))
            or not np.isfinite(float(val))
        ):
            raise DataValidationError(f"{name} must be a finite number, got {val!r}.")

    t1: float = float(theta_1)
    t2: float = float(theta_2)
    if t1 > t2:
        raise ThresholdCalibrationError(
            f"Policy requires theta_1 <= theta_2, got theta_1={t1}, theta_2={t2}."
        )

    risk_lambda_val: float = _validate_risk_lambda(risk_lambda)
    y_true_arr, y_prob_arr, eq_array = _validate_simulation_inputs(
        y_true=y_true,
        y_prob_calibrated=y_prob_calibrated,
        equipment_type=equipment_type,
    )
    realizations: dict[str, np.ndarray] = generate_cost_realizations(
        equipment_type=eq_array,
        n_iterations=n_iterations,
        random_seed=random_seed,
    )

    cost_iter: np.ndarray = _compute_trizone_cost_iterations(
        y_true_arr=y_true_arr,
        y_prob_arr=y_prob_arr,
        realizations=realizations,
        theta_1=t1,
        theta_2=t2,
    )

    expected_cost: float = float(np.mean(cost_iter))
    cost_std: float = float(np.std(cost_iter))
    p10_cost: float = float(np.percentile(cost_iter, 10))
    p90_cost: float = float(np.percentile(cost_iter, 90))
    p95_cost: float = float(np.percentile(cost_iter, 95))
    objective_value: float = float(expected_cost + (risk_lambda_val * p90_cost))

    return {
        "expected_cost": expected_cost,
        "cost_std": cost_std,
        "p10_cost": p10_cost,
        "p90_cost": p90_cost,
        "p95_cost": p95_cost,
        "objective_value": objective_value,
    }
