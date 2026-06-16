"""
Fast configuration preset for rapid VQE benchmarking.

Designed for quick iteration and testing. Uses H₂ in STO-3G (4 qubits with
Parity + two-qubit reduction), UCCSD with 1 repetition, and a loose convergence
tolerance with COBYLA (gradient-free, robust for noisy landscapes).

Typical wall-clock time:
    Noiseless (StatevectorEstimator) : ≈ 2 – 10 s
    Noisy     (AerEstimatorV2)       : ≈ 30 – 120 s

Usage in notebook Cell 1:
    from fast_config import EXPERIMENT
"""

EXPERIMENT = {
    # ── Molecule ────────────────────────────────────────────────────────────
    "molecule":        "H2",        # Smallest closed-shell molecule; 2 electrons
    "basis":           "sto-3g",    # Minimal basis: 1 spatial orbital per H → 2 total
    "charge":          0,
    "multiplicity":    1,           # Singlet (spin=0)

    # ── Space reduction (both OFF for full-space benchmark) ─────────────────
    "freeze_core":     False,       # H2 has no core electrons to freeze
    "use_active_space":False,
    "active_electrons":None,
    "active_orbitals": None,

    # ── Qubit mapping ────────────────────────────────────────────────────────
    "mapper":          "parity",    # Parity + two-qubit reduction → 2 qubits

    # ── Ansatz ───────────────────────────────────────────────────────────────
    "ansatz":          "uccsd",
    "reps":            1,
    "k":               1,

    # ── Optimiser ────────────────────────────────────────────────────────────
    "optimizer":       "cobyla",    # Gradient-free; good for noisy landscapes
    "maxiter":         200,
    "tol":             1e-4,        # Loose tolerance for speed

    # ── Initial point ─────────────────────────────────────────────────────
    "initial_point":   "hf",        # HF (zeros) – cheap and reliable

    # ── Estimator / simulation ───────────────────────────────────────────────
    "noise":           False,       # Switch to True for noisy simulation
    "shots":           2048,
    "seed":            42,
}
