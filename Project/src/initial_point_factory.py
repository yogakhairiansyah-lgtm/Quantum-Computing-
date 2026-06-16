"""
Initial point factory for VQE parameter initialisation.

Provides HF, MP2, zero, and random initial parameter vectors for VQE ansatze.
The HF and MP2 strategies use Qiskit Nature's built-in initial-point classes
where available, with graceful fallback to simpler alternatives.

Initial point semantics:

    HF (Hartree-Fock):
        All excitation amplitudes set to zero. For UCC ansatze, this recovers
        the Hartree-Fock state at θ=0. This is the standard starting point
        for UCCSD optimisations. For HEA circuits it simply returns zeros.

    MP2 (Second-order Møller-Plesset):
        T₂ amplitudes from MP2 perturbation theory are used to seed the
        double-excitation parameters. Singles are set to zero. Provides a
        better starting point than HF for systems with moderate correlation.
        Falls back to HF (zeros) if MP2 amplitudes are unavailable (e.g.,
        for HEA ansatze or when PySCF MP2 data is missing).

    Zero:
        All parameters set to zero. Equivalent to HF for UCC ansatze but
        named separately for clarity when used with HEA.

    Random:
        Uniform random in [−π, π]. Seeded for reproducibility.
"""

from __future__ import annotations

import warnings

import numpy as np
from qiskit.circuit import QuantumCircuit


def get_initial_point(
    name: str,
    ansatz: QuantumCircuit,
    problem=None,
    seed: int = 42,
) -> np.ndarray:
    """Build and return an initial parameter vector for VQE.

    Args:
        name: Initial point strategy (case-insensitive). Supported values:

            - ``'hf'`` or ``'hartree-fock'`` — HF reference (zeros for UCC).
            - ``'mp2'``                       — MP2-seeded amplitudes.
            - ``'zero'``                      — All zeros.
            - ``'random'``                    — Uniform random in [−π, π].

        ansatz: The parameterised ansatz circuit. Its ``num_parameters``
            attribute determines the vector length.
        problem: Optional :class:`ElectronicStructureProblem`. Required for
            MP2 initialisation. Ignored by Zero and Random strategies.
        seed: Random seed for the ``'random'`` strategy. Default: 42.

    Returns:
        A 1-D :class:`numpy.ndarray` of shape ``(ansatz.num_parameters,)``
        containing the initial parameter values.

    Raises:
        ValueError: If ``name`` is not a recognised strategy.

    Warns:
        UserWarning: When MP2 initialisation is unavailable and a fallback
            is used.

    Example:
        >>> point = get_initial_point("hf", ansatz)
        >>> point = get_initial_point("mp2", ansatz, problem=problem)
        >>> point = get_initial_point("random", ansatz, seed=123)
    """
    key = name.lower().strip().replace("-", "").replace("_", "").replace(" ", "")

    if key == "zero":
        return _zero_point(ansatz)

    if key in ("hf", "hartreefock"):
        return _hf_point(ansatz)

    if key == "mp2":
        return _mp2_point(ansatz, problem)

    if key in ("random", "rand"):
        return _random_point(ansatz, seed=seed)

    supported = ["zero", "hf", "mp2", "random"]
    raise ValueError(
        f"Initial point strategy '{name}' is not supported.\n"
        f"Supported options: {supported}"
    )


# ── Private builders ─────────────────────────────────────────────────────────

def _zero_point(ansatz: QuantumCircuit) -> np.ndarray:
    """Return a zero vector matching the ansatz parameter count.

    Args:
        ansatz: Parameterised circuit.

    Returns:
        Zero numpy array of length ``ansatz.num_parameters``.
    """
    return np.zeros(ansatz.num_parameters)


def _hf_point(ansatz: QuantumCircuit) -> np.ndarray:
    """Return the Hartree-Fock initial point (zeros for UCC ansatze).

    Attempts to use :class:`qiskit_nature.second_q.algorithms.initial_points
    .HFInitialPoint`. Falls back to zeros if unavailable.

    Args:
        ansatz: Parameterised circuit.

    Returns:
        Numpy array of HF initial amplitudes (zeros for UCC).
    """
    try:
        from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
        hf = HFInitialPoint()
        hf.ansatz = ansatz
        return hf.to_numpy_array()
    except Exception:
        # HF initial point for UCC ansatz is always zeros
        return np.zeros(ansatz.num_parameters)


def _mp2_point(ansatz: QuantumCircuit, problem) -> np.ndarray:
    """MP2 amplitudes via PySCF langsung — bypass qiskit_nature algorithms."""
    try:
        from pyscf import gto, scf, mp as pyscf_mp

        mol_info = problem.molecule

        # problem.basis adalah ElectronicBasis enum, bukan string
        # ambil basis string dari hamiltonian atau hardcode dari EXPERIMENT
        try:
            basis_str = problem.hamiltonian.basis.value  # 'sto-3g', 'cc-pvdz', dll
        except Exception:
            basis_str = "sto-3g"   # fallback hardcode

        atom_list = [
            (sym, tuple(coord))
            for sym, coord in zip(mol_info.symbols, mol_info.coords)
        ]

        pyscf_mol         = gto.Mole()
        pyscf_mol.atom    = atom_list
        pyscf_mol.basis   = basis_str
        pyscf_mol.charge  = mol_info.charge
        pyscf_mol.spin    = mol_info.multiplicity - 1
        pyscf_mol.unit    = "Angstrom"
        pyscf_mol.verbose = 0
        pyscf_mol.build()

        mf      = scf.RHF(pyscf_mol).run(verbose=0)
        mp2_obj = pyscf_mp.MP2(mf).run(verbose=0)
        t2      = mp2_obj.t2

        amps    = t2.flatten()
        nparams = ansatz.num_parameters
        out     = np.zeros(nparams)
        out[:min(len(amps), nparams)] = amps[:nparams]
        print(f"  MP2 OK: {len(amps)} amplitudes → {nparams} params")
        return out

    except Exception as exc:
        # Cek nilai basis
        print(f"DEBUG basis type  : {type(problem.basis)}")
        print(f"DEBUG basis value : {problem.basis}")
        print(f"DEBUG basis dir   : {[a for a in dir(problem.basis) if not a.startswith('_')]}")
        warnings.warn(
            f"MP2 via PySCF gagal: {exc}. Fallback ke zeros.",
            UserWarning,
            stacklevel=3,
        )
        return np.zeros(ansatz.num_parameters)

def _random_point(ansatz: QuantumCircuit, seed: int = 42) -> np.ndarray:
    """Return a uniformly random initial parameter vector in [−π, π].

    Args:
        ansatz: Parameterised circuit.
        seed: Integer random seed for reproducibility.

    Returns:
        Numpy array of random values in [−π, π].
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(-np.pi, np.pi, ansatz.num_parameters)


def list_supported_initial_points() -> list[str]:
    """Return a sorted list of all supported initial point strategy names.

    Returns:
        Sorted list of supported strategy names.
    """
    return ["hf", "mp2", "random", "zero"]
