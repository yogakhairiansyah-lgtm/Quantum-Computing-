"""
Optimizer factory for VQE benchmark experiments.

Provides a unified interface for constructing Qiskit Algorithms classical
optimizers compatible with Qiskit 2.3.0 and qiskit-algorithms >= 0.3.0.

Tolerance parameter mapping (each optimizer has a different convergence API):

    COBYLA  : ``tol`` keyword — function-value convergence tolerance.
    SLSQP   : ``ftol`` keyword — function-value tolerance (projected gradient
              threshold in SciPy's SLSQP backend).
    L_BFGS_B: ``ftol`` and ``gtol`` keywords — function-value and projected
              gradient convergence. Both are set to ``tol``.
    SPSA    : Stochastic approximation; no meaningful gradient tolerance.
              Only ``maxiter`` is applied; ``tol`` is stored but not forwarded.
"""

from __future__ import annotations

from qiskit_algorithms.optimizers import (
    COBYLA,
    L_BFGS_B,
    SLSQP,
    SPSA,
    Optimizer,
)


# ── Registry ─────────────────────────────────────────────────────────────────
_SUPPORTED_OPTIMIZERS: dict[str, str] = {
    "cobyla":    "cobyla",
    "slsqp":     "slsqp",
    "l_bfgs_b":  "lbfgsb",
    "lbfgsb":    "lbfgsb",
    "l-bfgs-b":  "lbfgsb",
    "l_bfgs_b":  "lbfgsb",
    "l bfgs b":  "lbfgsb",
    "spsa":      "spsa",
}


def get_optimizer(
    name: str,
    maxiter: int = 500,
    tol: float = 1e-6,
) -> Optimizer:
    """Construct and return a Qiskit Algorithms classical optimizer.

    Tolerance values are mapped to each optimizer's native convergence
    parameter. See the module docstring for details.

    Args:
        name: Optimizer name (case-insensitive). Supported values:

            - ``'cobyla'``            → :class:`COBYLA`
            - ``'slsqp'``             → :class:`SLSQP`
            - ``'l_bfgs_b'`` / ``'lbfgsb'`` / ``'l-bfgs-b'``
                                       → :class:`L_BFGS_B`
            - ``'spsa'``              → :class:`SPSA`

        maxiter: Maximum number of function evaluations (COBYLA, L_BFGS_B)
            or iterations (SLSQP, SPSA).
        tol: Convergence tolerance. Mapped to optimizer-specific parameter
            names. Ignored for SPSA.

    Returns:
        A fully initialised :class:`Optimizer` instance.

    Raises:
        ValueError: If ``name`` is not a supported optimizer identifier.

    Example:
        >>> opt = get_optimizer("slsqp", maxiter=300, tol=1e-8)
        >>> opt = get_optimizer("cobyla", maxiter=1000)
        >>> opt = get_optimizer("l-bfgs-b", maxiter=500, tol=1e-9)
    """
    key = name.lower().strip().replace("-", "_").replace(" ", "_")

    # Normalise common variants
    lookup_key = key.replace("l_bfgs_b", "lbfgsb").replace("lbfgsb", "lbfgsb")

    if lookup_key not in set(_SUPPORTED_OPTIMIZERS.values()):
        # Also try direct registry look-up
        raw_key = name.lower().strip()
        if raw_key not in _SUPPORTED_OPTIMIZERS:
            supported = sorted(set(_SUPPORTED_OPTIMIZERS.keys()))
            raise ValueError(
                f"Optimizer '{name}' is not supported.\n"
                f"Supported optimizers: {supported}"
            )
        lookup_key = _SUPPORTED_OPTIMIZERS[raw_key]

    if lookup_key == "cobyla":
        return COBYLA(maxiter=maxiter, tol=tol)

    if lookup_key == "slsqp":
        # ftol harus masuk ke options dict, bukan kwarg langsung
        return SLSQP(
            maxiter=maxiter,
            options={"ftol": tol, "maxiter": maxiter},
        )

    if lookup_key == "lbfgsb":
        # gtol dan ftol harus masuk ke options dict (scipy >= 1.11)
        # L-BFGS-B konvensi: ftol = gtol * 1e7 (scipy default ratio)
        return L_BFGS_B(
            options={
                "maxfun":  maxiter,
                "maxiter": maxiter,
                "gtol":    tol,
                "ftol":    tol * 1e7,
            }
        )

    if lookup_key == "spsa":
        return SPSA(maxiter=maxiter)
    
    # Fallback (should be unreachable)
    raise RuntimeError(
        f"Optimizer '{name}' passed validation but has no constructor. "
        "Please report this as a bug."
    )


def list_supported_optimizers() -> list[str]:
    """Return a sorted list of all supported optimizer name strings.

    Returns:
        Sorted list of canonical names and aliases.

    Example:
        >>> list_supported_optimizers()
        ['cobyla', 'l-bfgs-b', 'l_bfgs_b', 'lbfgsb', 'slsqp', 'spsa']
    """
    return sorted(set(_SUPPORTED_OPTIMIZERS.keys()))
