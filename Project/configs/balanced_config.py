"""
Balanced configuration preset for H₂O VQE benchmarking.

A representative research configuration: H₂O in STO-3G, full space,
UCCSD ansatz with Parity mapping (two-qubit reduction, 12 qubits),
SLSQP optimizer, and HF initialisation. Targets chemical accuracy
(< 1 mHa from FCI) within a reasonable compute budget.

System characteristics (STO-3G, full space):
    Electrons          : 10 (5α + 5β)
    Spatial orbitals   : 7
    Spin orbitals      : 14
    Qubits (Parity+2qr): 12
    UCCSD parameters   : ~26 (reps=1)

Typical wall-clock time:
    Noiseless (StatevectorEstimator) : ≈ 20 – 120 s
    Noisy     (AerEstimatorV2)       : ≈ 5 – 30 min

Usage in notebook Cell 1:
    from balanced_config import EXPERIMENT
"""

EXPERIMENT = {
    # ── Molecule ────────────────────────────────────────────────────────────
    "molecule":        "H2O",
    "basis":           "sto-3g",
    "charge":          0,
    "multiplicity":    1,

    # ── Space reduction ──────────────────────────────────────────────────────
    "freeze_core":     False,       # Full-space benchmark
    "use_active_space":False,
    "active_electrons":None,
    "active_orbitals": None,

    # ── Qubit mapping ────────────────────────────────────────────────────────
    "mapper":          "parity",

    # ── Ansatz ───────────────────────────────────────────────────────────────
    "ansatz":          "uccsd",
    "reps":            1,           # Single UCCSD layer
    "k":               1,

    # ── Optimiser ────────────────────────────────────────────────────────────
    "optimizer":       "slsqp",     # Gradient-based; fast convergence noiseless
    "maxiter":         500,
    "tol":             1e-6,

    # ── Initial point ─────────────────────────────────────────────────────
    "initial_point":   "hf",

    # ── Estimator / simulation ───────────────────────────────────────────────
    "noise":           False,
    "shots":           4096,
    "seed":            123,
}
