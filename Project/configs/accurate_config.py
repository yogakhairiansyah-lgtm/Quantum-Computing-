"""
Accurate configuration preset for high-fidelity H₂O benchmarking.

Uses UCCGSD (generalised excitations) with L-BFGS-B, tight tolerances,
and Freeze Core to reduce the orbital space. Targets sub-mHa accuracy
and is suitable for publishing-quality benchmarks.

System characteristics (STO-3G, freeze core):
    Full-space electrons    : 10 (5α + 5β)
    After freeze core       : 8 electrons, 6 spatial orbitals
    Qubits (Parity+2qr)     : 10
    UCCGSD parameters       : larger than UCCSD (all occ–occ, vir–vir pairs)

Typical wall-clock time:
    Noiseless (StatevectorEstimator) : ≈ 2 – 15 min
    Noisy     (AerEstimatorV2)       : hours (consider reducing shots)

Benchmark variants covered by this config:
    1. Full Space    — set freeze_core=False, use_active_space=False
    2. Freeze Core   — set freeze_core=True  (default below)
    3. Active Space  — set use_active_space=True, active_electrons=4,
                       active_orbitals=4
    4. FC + AS       — combine both transformers

Usage in notebook Cell 1:
    from accurate_config import EXPERIMENT
"""

EXPERIMENT = {
    # ── Molecule ────────────────────────────────────────────────────────────
    "molecule":        "H2O",
    "basis":           "sto-3g",
    "charge":          0,
    "multiplicity":    1,

    # ── Space reduction ──────────────────────────────────────────────────────
    "freeze_core":     True,        # Freeze 1s core of oxygen (2 electrons)
    "use_active_space":False,       # Set True to further restrict orbitals
    "active_electrons":4,           # Used only when use_active_space=True
    "active_orbitals": 4,           # Used only when use_active_space=True

    # ── Qubit mapping ────────────────────────────────────────────────────────
    "mapper":          "parity",

    # ── Ansatz ───────────────────────────────────────────────────────────────
    "ansatz":          "uccgsd",    # Generalised SD — higher accuracy than UCCSD
    "reps":            1,
    "k":               1,

    # ── Optimiser ────────────────────────────────────────────────────────────
    "optimizer":       "l_bfgs_b",  # Quasi-Newton; excellent for noiseless VQE
    "maxiter":         1000,
    "tol":             1e-9,        # Tight convergence for publication quality

    # ── Initial point ─────────────────────────────────────────────────────
    "initial_point":   "mp2",       # MP2 seeds provide better UCCGSD starting pt

    # ── Estimator / simulation ───────────────────────────────────────────────
    "noise":           False,
    "shots":           8192,
    "seed":            123,
}
