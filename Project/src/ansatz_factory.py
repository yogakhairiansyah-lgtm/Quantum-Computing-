"""
Ansatz factory for VQE molecular quantum chemistry benchmarks.

Provides a unified interface for constructing UCCSD, UCCGSD, k-UpCCGSD, and
Hardware-Efficient Ansatz (HEA) circuits compatible with Qiskit Nature 0.7+
and Qiskit 2.3.0.

Notes on k-UpCCGSD:
    The k-UpCCGSD ansatz (Lee et al., J. Chem. Theory Comput. 2019) is
    implemented via Qiskit Nature's :class:`PUCCD` with ``include_singles=True``
    and ``reps=k``. This captures the paired (geminal) double excitations plus
    generalised singles repeated k times, which is the defining structure of
    the k-UpCCGSD family. The ``reps=k`` parameter controls the number of
    product-formula layers.

Notes on HEA:
    Hardware-Efficient Ansatz circuits use gate-based parametrised rotations
    without chemistry-specific structure. They are useful as lower-cost
    baselines. The qubit count is inferred from the problem and mapper
    without constructing the full Hamiltonian.
"""

from __future__ import annotations

import warnings
from typing import Literal

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem


# ── Supported ansatz names ───────────────────────────────────────────────────
_SUPPORTED_ANSATZE = {
    "uccsd":     "UCCSD",
    "uccgsd":    "UCCGSD",
    "k-upccgsd": "k-UpCCGSD",
    "kupccgsd":  "k-UpCCGSD",
    "k_upccgsd": "k-UpCCGSD",
    "hea":       "HEA",
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_num_qubits(
    problem: ElectronicStructureProblem,
    mapper: QubitMapper,
) -> int:
    """Estimate qubit count from problem and mapper without building Hamiltonian.

    Args:
        problem: Electronic structure problem.
        mapper: Qubit mapper.

    Returns:
        Expected number of qubits after mapping.
    """
    from qiskit_nature.second_q.mappers import ParityMapper

    num_spin_orbitals = 2 * problem.num_spatial_orbitals
    if isinstance(mapper, ParityMapper) and mapper.num_particles is not None:
        return num_spin_orbitals - 2
    return num_spin_orbitals


def _build_hf_initial_state(
    problem: ElectronicStructureProblem,
    mapper: QubitMapper,
) -> HartreeFock:
    """Construct the Hartree-Fock reference state circuit.

    Args:
        problem: Electronic structure problem.
        mapper: Qubit mapper.

    Returns:
        A :class:`HartreeFock` circuit to prepend to the ansatz.
    """
    return HartreeFock(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def get_ansatz(
    name: str,
    problem: ElectronicStructureProblem,
    mapper: QubitMapper,
    reps: int = 1,
    k: int = 1,
    hea_type: str = "efficient_su2",
    entanglement: str = "linear",
) -> QuantumCircuit:
    """Build and return a parameterised ansatz circuit.

    UCC-family circuits (UCCSD, UCCGSD, k-UpCCGSD) are initialised with a
    Hartree-Fock state constructed from ``problem`` and ``mapper``.
    HEA circuits require only the qubit count, which is derived from the
    problem and mapper without computing the full Hamiltonian.

    Args:
        name: Ansatz identifier (case-insensitive). Supported values:

            - ``'uccsd'``      — Unitary Coupled-Cluster Singles and Doubles.
            - ``'uccgsd'``     — Unitary CC Generalised Singles and Doubles.
            - ``'k-upccgsd'`` or ``'kupccgsd'`` — k-UpCCGSD (paired UCC,
              ``reps=k``). See module docstring.
            - ``'hea'``        — Hardware-Efficient Ansatz (type via
              ``hea_type``).

        problem: Electronic structure problem providing particle number and
            orbital count.
        mapper: Qubit mapper for the fermionic-to-qubit transformation. Must
            be the **same** mapper used to build the qubit Hamiltonian.
        reps: Number of repetition layers for UCCSD/UCCGSD/HEA. For
            k-UpCCGSD, use ``k`` instead; ``reps`` is ignored.
        k: Number of k-UpCCGSD layers (the k in Lee et al. 2019).
            Ignored for all other ansatze.
        hea_type: HEA block architecture. One of:

            - ``'efficient_su2'`` (default) — SU(2) + CNOT layers.
            - ``'realamplitudes'``           — Ry + CNOT layers.

        entanglement: Entanglement pattern passed to HEA circuits.
            Supported: ``'linear'``, ``'full'``, ``'circular'``, ``'sca'``.
            Default is ``'linear'``.

    Returns:
        A fully-parameterised :class:`QuantumCircuit` for use with VQE.

    Raises:
        ValueError: If ``name`` or ``hea_type`` is not recognised.

    Example:
        >>> ansatz = get_ansatz("uccsd", problem, mapper, reps=1)
        >>> ansatz = get_ansatz("hea", problem, mapper, reps=3, hea_type="realamplitudes")
        >>> ansatz = get_ansatz("k-upccgsd", problem, mapper, k=2)
    """
    key = name.lower().strip().replace("-", "").replace("_", "")

    if key == "uccsd":
        return _build_uccsd(problem, mapper, reps=reps)

    if key == "uccgsd":
        return _build_uccgsd(problem, mapper, reps=reps)

    if key in ("kupccgsd", "kupcgsd"):
        return _build_kupccgsd(problem, mapper, k=k)

    if key == "hea":
        return _build_hea(
            problem, mapper,
            reps=reps,
            hea_type=hea_type,
            entanglement=entanglement,
        )

    supported = list(_SUPPORTED_ANSATZE.keys())
    raise ValueError(
        f"Ansatz '{name}' is not supported.\n"
        f"Supported ansatze: {supported}"
    )


# ── Concrete builders ────────────────────────────────────────────────────────

def _build_uccsd(
    problem: ElectronicStructureProblem,
    mapper: QubitMapper,
    reps: int = 1,
) -> QuantumCircuit:
    """Construct a UCCSD ansatz with Hartree-Fock initial state.

    Args:
        problem: Electronic structure problem.
        mapper: Qubit mapper.
        reps: Number of repetition layers.

    Returns:
        Parameterised UCCSD circuit.
    """
    initial_state = _build_hf_initial_state(problem, mapper)
    return UCCSD(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
        reps=reps,
    )


def _build_uccgsd(
    problem: ElectronicStructureProblem,
    mapper: QubitMapper,
    reps: int = 1,
) -> QuantumCircuit:
    """Construct a UCCGSD ansatz with Hartree-Fock initial state.

    UCCGSD uses generalised excitations (all orbital pairs, not only
    occupied–virtual), increasing expressibility at the cost of more
    parameters compared to UCCSD.

    Args:
        problem: Electronic structure problem.
        mapper: Qubit mapper.
        reps: Number of repetition layers.

    Returns:
        Parameterised UCCGSD circuit.
    """
    initial_state = _build_hf_initial_state(problem, mapper)

    # Prefer the dedicated UCCGSD class; fall back to UCCSD(generalized=True).
    try:
        from qiskit_nature.second_q.circuit.library import UCCGSD
        return UCCGSD(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
            initial_state=initial_state,
            reps=reps,
        )
    except ImportError:
        warnings.warn(
            "UCCGSD class not found; falling back to UCCSD(generalized=True).",
            UserWarning,
            stacklevel=3,
        )
        return UCCSD(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
            initial_state=initial_state,
            reps=reps,
            generalized=True,
        )


def _build_kupccgsd(
    problem: ElectronicStructureProblem,
    mapper: QubitMapper,
    k: int = 1,
) -> QuantumCircuit:
    """Construct a k-UpCCGSD ansatz via PUCCD with singles and reps=k.

    Implements the k-UpCCGSD ansatz from Lee et al. (J. Chem. Theory Comput.
    2019) using :class:`PUCCD` with ``include_singles=True`` and ``reps=k``.
    Paired doubles enforce the geminal (same-spatial-orbital) structure, and
    singles add the generalised S component (UpCCGSD vs UpCCGD).

    For open-shell systems (α ≠ β electrons), a fallback to UCCGSD is used
    because paired doubles are defined only for equal spin-sector occupancies.

    Args:
        problem: Electronic structure problem.
        mapper: Qubit mapper.
        k: Number of k-UpCCGSD layers (the k in Lee et al. 2019).

    Returns:
        Parameterised k-UpCCGSD circuit (PUCCD with singles, reps=k).
    """
    num_alpha, num_beta = problem.num_particles
    initial_state = _build_hf_initial_state(problem, mapper)

    if num_alpha != num_beta:
        warnings.warn(
            f"k-UpCCGSD (PUCCD) is designed for closed-shell systems with "
            f"equal α/β electrons. Got ({num_alpha}, {num_beta}). "
            f"Falling back to UCCGSD with reps={k}.",
            UserWarning,
            stacklevel=3,
        )
        return _build_uccgsd(problem, mapper, reps=k)

    try:
        from qiskit_nature.second_q.circuit.library import PUCCD
        return PUCCD(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
            initial_state=initial_state,
            include_singles=[True, True],  # [alpha_singles, beta_singles]
            reps=k,                        # jumlah layer k-UpCCGSD
        )
    except (ImportError, TypeError) as exc:
        warnings.warn(
            f"PUCCD unavailable or incompatible ({exc}). "
            f"Falling back to UCCGSD with reps={k}.",
            UserWarning,
            stacklevel=3,
        )
        return _build_uccgsd(problem, mapper, reps=k)


def _build_hea(
    problem: ElectronicStructureProblem,
    mapper: QubitMapper,
    reps: int = 3,
    hea_type: str = "efficient_su2",
    entanglement: str = "linear",
) -> QuantumCircuit:
    """Construct a Hardware-Efficient Ansatz (HEA) circuit.

    HEA circuits use parameterised rotations and entangling layers without
    chemistry-specific structure. The qubit count is derived from the problem
    and mapper without computing the full Hamiltonian.

    Args:
        problem: Electronic structure problem (used to infer qubit count).
        mapper: Qubit mapper (used to infer qubit count).
        reps: Number of rotation-entanglement repetition layers.
        hea_type: Architecture variant:

            - ``'efficient_su2'`` — SU(2) rotations + CNOT entanglement.
            - ``'realamplitudes'`` — Ry rotations + CNOT entanglement.

        entanglement: Entanglement pattern. Supported: ``'linear'``,
            ``'full'``, ``'circular'``, ``'sca'``.

    Returns:
        Parameterised HEA :class:`QuantumCircuit`.

    Raises:
        ValueError: If ``hea_type`` is not recognised.
    """
    num_qubits = _get_num_qubits(problem, mapper)
    hea_key = hea_type.lower().strip().replace("_", "").replace("-", "")

    if hea_key in ("efficientsu2", "su2"):
        return EfficientSU2(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
        )

    if hea_key == "realamplitudes":
        return RealAmplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
        )

    raise ValueError(
        f"HEA type '{hea_type}' is not recognised. "
        f"Supported: 'efficient_su2', 'realamplitudes'."
    )


def list_supported_ansatze() -> list[str]:
    """Return a sorted list of all recognised ansatz name strings.

    Returns:
        List of canonical names and aliases.

    Example:
        >>> list_supported_ansatze()
        ['hea', 'k-upccgsd', 'k_upccgsd', 'kupccgsd', 'uccgsd', 'uccsd']
    """
    return sorted(_SUPPORTED_ANSATZE.keys())
