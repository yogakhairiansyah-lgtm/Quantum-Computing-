"""
ADAPT-VQE factory for Qiskit Nature + Qiskit Algorithms.

ADAPT-VQE (Grimsley et al., Nature Commun. 2019) adaptively grows an ansatz
from a pool of fermionic excitation operators. At each macro-iteration it
selects the operator with the largest energy gradient, appends it to the
circuit, and re-optimises all parameters with a standard VQE step.

This module is intentionally separate from ``ansatz_factory.py`` because
ADAPT-VQE does not produce a fixed-structure circuit — the ansatz is
constructed dynamically during the optimisation. The notebook therefore
branches on ``IS_ADAPT_VQE`` to route between this builder and the standard
one.

Compatible with qiskit-algorithms >= 0.3.0 (V2 primitive interface) and
Qiskit 2.3.0.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
from qiskit_algorithms.optimizers import Optimizer
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem


def build_adapt_vqe(
    estimator,
    optimizer: Optimizer,
    problem: ElectronicStructureProblem,
    mapper: QubitMapper,
    threshold: float = 1e-5,
    max_iterations: int = 20,
) -> tuple[AdaptVQE, QuantumCircuit]:
    """Construct an ADAPT-VQE algorithm object ready for execution.

    ADAPT-VQE uses a UCCSD operator pool as the candidate excitation library.
    At each macro-iteration the operator with the largest gradient magnitude is
    appended to the growing ansatz, and the inner VQE re-optimises all current
    parameters. The algorithm terminates when every remaining gradient falls
    below ``threshold`` or ``max_iterations`` iterations are reached.

    Construction steps performed here:

    1. **Hartree-Fock initial state** — the reference circuit is prepended to
       every grown ansatz.
    2. **UCCSD operator pool** — all single and double excitation operators
       from the molecular orbital space are used as candidates.
    3. **Base VQE solver** — a standard :class:`VQE` instance using the
       provided estimator and optimizer; ADAPT-VQE calls this sub-solver after
       each operator addition.
    4. **AdaptVQE wrapper** — sets convergence and iteration limits.

    Args:
        estimator: A V2-compatible estimator primitive
            (``StatevectorEstimator`` or ``AerEstimatorV2``). Gradient-based
            optimizers work best for ADAPT-VQE inner loops.
        optimizer: Qiskit Algorithms optimizer for the inner VQE loop.
            Gradient-based choices (SLSQP, L_BFGS_B) are strongly recommended
            for fast convergence of the inner optimisation.
        problem: Electronic structure problem defining the molecular Hilbert
            space (particle number and orbital count). Must already have any
            Freeze Core / Active Space transformations applied.
        mapper: Qubit mapper for the fermionic-to-qubit transformation.
            Must match the mapper used when building the qubit Hamiltonian
            in the notebook.
        threshold: Convergence threshold for operator gradient norms (‖∂E/∂θ‖).
            The outer ADAPT loop stops when all gradients satisfy
            ``‖g‖ < threshold``. Default: 1e-5.
        max_iterations: Maximum number of ADAPT macro-iterations (operator
            additions). Default: 20.

    Returns:
        A two-element tuple ``(adapt_vqe, pool_ansatz)`` where:

        - ``adapt_vqe`` (:class:`AdaptVQE`) — configured algorithm, call
          ``adapt_vqe.compute_minimum_eigenvalue(qubit_op)`` to run.
        - ``pool_ansatz`` (:class:`QuantumCircuit`) — the UCCSD operator-pool
          ansatz that defines the excitation library (not the grown circuit).
          Returned for circuit metric display in the notebook.

    Example:
        >>> adapt_vqe, pool_ansatz = build_adapt_vqe(
        ...     estimator=estimator,
        ...     optimizer=SLSQP(maxiter=300),
        ...     problem=problem,
        ...     mapper=mapper,
        ...     threshold=1e-5,
        ...     max_iterations=15,
        ... )
        >>> result = adapt_vqe.compute_minimum_eigenvalue(qubit_hamiltonian)
        >>> energy = result.eigenvalue.real
    """
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_particles = problem.num_particles

    # ── Step 1: Hartree-Fock initial state ──────────────────────────────────
    initial_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )

    # ── Step 2: UCCSD operator pool ─────────────────────────────────────────
    # ADAPT-VQE selects operators from this pool greedily at each step.
    # The grown ansatz is a sub-sequence of pool operators chosen by gradient.
    pool_ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
        reps=1,
    )

    # ── Step 3: Inner VQE solver ─────────────────────────────────────────────
    # ADAPT-VQE calls this sub-solver after each operator is added.
    # The callback set by the notebook will be attached to this solver.
    base_vqe = VQE(
        estimator=estimator,
        ansatz=pool_ansatz,
        optimizer=optimizer,
    )

    # ── Step 4: AdaptVQE wrapper ─────────────────────────────────────────────
    adapt_vqe = AdaptVQE(solver=base_vqe)
    adapt_vqe.threshold = threshold
    adapt_vqe.max_iterations = max_iterations

    return adapt_vqe, pool_ansatz
