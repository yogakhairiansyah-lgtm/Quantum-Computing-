"""
Analysis utilities for VQE benchmark experiments.

Provides energy error computation, circuit resource metrics, problem-size
metrics, and a structured pandas DataFrame summarising a complete benchmark
run. All functions are stateless and accept primitive Python / NumPy / Qiskit
types for easy integration with the notebook workflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit

# ── PES summary table ────────────────────────────────────────────────────────

def create_pes_summary_table(
    experiment: dict,
    pes_results: list[dict],
) -> pd.DataFrame:
    rows = []
    for p in pes_results:
        err_ha = abs(p["vqe_energy"] - p["fci_energy"])
        rows.append({
            "Bond Length (Å)":  round(p["r"], 6),
            "VQE Energy (Ha)":  round(p["vqe_energy"], 8),
            "FCI Energy (Ha)":  round(p["fci_energy"], 8),
            "Error (Ha)":       round(err_ha, 8),
            "Error (mHa)":      round(p.get("error_mha", err_ha * 1000.0), 6),
            "Runtime (s)":      round(p.get("runtime", 0.0), 3),
            "Evals":            int(p.get("n_evals", 0)),
            "Molecule":         experiment.get("molecule", "Unknown"),
            "Basis":            experiment.get("basis", "Unknown"),
            "Ansatz":           experiment.get("ansatz", "Unknown"),
            "Mapper":           experiment.get("mapper", "Unknown"),
            "Optimizer":        experiment.get("optimizer", "Unknown"),
            "Freeze Core":      experiment.get("freeze_core", False),
            "Active Space":     experiment.get("use_active_space", False),
            "Scan Type":        experiment.get("scan_type", "N/A"),
        })
    return pd.DataFrame(rows)

# ── Energy ───────────────────────────────────────────────────────────────────

def compute_energy_error(
    vqe_energy: float,
    reference_energy: float,
) -> dict[str, float]:
    """Compute the absolute energy error of a VQE result.

    Args:
        vqe_energy: Final VQE energy in Hartree.
        reference_energy: Reference (e.g. FCI) energy in Hartree.

    Returns:
        Dictionary with keys:

        - ``'hartree_error'``   — |E_VQE − E_ref| in Hartree.
        - ``'millihart_error'`` — same error in milli-Hartree (mHa).
        - ``'vqe_energy'``      — the VQE energy (Ha).
        - ``'reference_energy'`` — the reference energy (Ha).

    Example:
        >>> errs = compute_energy_error(-75.012345, -75.013500)
        >>> errs["millihart_error"]
        1.155
    """
    delta = abs(float(vqe_energy) - float(reference_energy))
    return {
        "hartree_error":    delta,
        "millihart_error":  delta * 1_000.0,
        "vqe_energy":       float(vqe_energy),
        "reference_energy": float(reference_energy),
    }


# ── Timing ───────────────────────────────────────────────────────────────────

def compute_runtime(start_time: float, end_time: float) -> float:
    """Return the elapsed wall-clock time between two timestamps.

    Args:
        start_time: Start timestamp from ``time.perf_counter()``.
        end_time: End timestamp from ``time.perf_counter()``.

    Returns:
        Elapsed time in seconds.

    Example:
        >>> import time
        >>> t0 = time.perf_counter()
        >>> time.sleep(0.1)
        >>> compute_runtime(t0, time.perf_counter())  # ≈ 0.1
    """
    return float(end_time) - float(start_time)


# ── Circuit metrics ──────────────────────────────────────────────────────────

def get_circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    """Extract resource metrics from a (possibly parameterised) quantum circuit.

    Attempts a single level of :meth:`QuantumCircuit.decompose` to reveal
    sub-gate structure before counting operations. Falls back to the original
    circuit if decomposition fails (e.g. the circuit has not yet been fully
    constructed by Qiskit Nature's lazy builder).

    Two-qubit gate count covers the most common native 2Q gates:
    ``cx``, ``ecr``, ``cz``, ``swap``, ``iswap``.

    Args:
        circuit: A parameterised or bound :class:`QuantumCircuit`.

    Returns:
        Dictionary with keys:

        - ``'depth'``         — circuit depth after one decomposition pass.
        - ``'num_parameters'`` — number of free parameters.
        - ``'gate_counts'``   — ``{gate_name: count}`` dict.
        - ``'cnot_count'``    — total count of 2-qubit entangling gates.

    Example:
        >>> metrics = get_circuit_metrics(ansatz)
        >>> metrics["num_parameters"]
        26
    """
    num_parameters = circuit.num_parameters

    # Try a single decomposition pass for more detailed gate counts
    try:
        decomposed = circuit.decompose()
        ops = dict(decomposed.count_ops())
        depth = decomposed.depth()
    except Exception:
        try:
            ops = dict(circuit.count_ops())
            depth = circuit.depth()
        except Exception:
            ops = {}
            depth = -1  # Unknown — circuit not yet built

    # Two-qubit entangling gate count
    two_qubit_gates = {"cx", "ecr", "cz", "swap", "iswap", "xx_minus_yy", "xx_plus_yy"}
    cnot_count = sum(ops.get(g, 0) for g in two_qubit_gates)

    return {
        "depth":          depth,
        "num_parameters": num_parameters,
        "gate_counts":    ops,
        "cnot_count":     cnot_count,
    }


# ── Problem metrics ──────────────────────────────────────────────────────────

def get_problem_metrics(
    problem,
    mapper=None,
) -> dict[str, Any]:
    """Extract system-size metrics from an electronic structure problem.

    Args:
        problem: :class:`qiskit_nature.second_q.problems.ElectronicStructureProblem`
            (possibly after Freeze Core / Active Space transformation).
        mapper: Optional qubit mapper. If provided and is a
            :class:`ParityMapper` with ``num_particles`` set, the qubit
            count is reduced by 2 (two-qubit reduction).

    Returns:
        Dictionary with keys:

        - ``'num_particles'``        — ``(n_alpha, n_beta)`` tuple.
        - ``'num_spatial_orbitals'`` — number of spatial (molecular) orbitals.
        - ``'num_spin_orbitals'``    — 2 × spatial orbitals.
        - ``'num_qubits'``           — mapped qubit count (or ``None`` if no
          mapper given).

    Example:
        >>> m = get_problem_metrics(problem, mapper=mapper)
        >>> m["num_qubits"]
        12
    """
    num_alpha, num_beta = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_spin_orbitals = 2 * num_spatial_orbitals

    num_qubits = None
    if mapper is not None:
        try:
            from qiskit_nature.second_q.mappers import ParityMapper
            if isinstance(mapper, ParityMapper) and mapper.num_particles is not None:
                num_qubits = num_spin_orbitals - 2
            else:
                num_qubits = num_spin_orbitals
        except Exception:
            num_qubits = num_spin_orbitals

    return {
        "num_particles":        (num_alpha, num_beta),
        "num_spatial_orbitals": num_spatial_orbitals,
        "num_spin_orbitals":    num_spin_orbitals,
        "num_qubits":           num_qubits,
    }


# ── Summary table ────────────────────────────────────────────────────────────

def create_summary_table(
    experiment: dict[str, Any],
    vqe_energy: float,
    reference_energy: float,
    runtime: float,
    num_evaluations: int,
    circuit_metrics: dict[str, Any],
    problem_metrics: dict[str, Any],
    energy_errors: dict[str, float],
) -> pd.DataFrame:
    """Create a pandas DataFrame summarising a single VQE benchmark run.

    The returned single-row DataFrame can be concatenated with results from
    other runs (different mappers, ansatze, optimizers) to build multi-run
    comparison tables.

    Args:
        experiment: The ``EXPERIMENT`` configuration dictionary from the
            notebook Cell 1.
        vqe_energy: Final optimised VQE energy in Hartree.
        reference_energy: Reference (FCI) energy in Hartree.
        runtime: Wall-clock execution time in seconds.
        num_evaluations: Total number of cost-function evaluations.
        circuit_metrics: Output of :func:`get_circuit_metrics`.
        problem_metrics: Output of :func:`get_problem_metrics`.
        energy_errors: Output of :func:`compute_energy_error`.

    Returns:
        A one-row :class:`pandas.DataFrame` with the columns listed below.

    Columns:
        Molecule, Basis, Freeze Core, Active Space, Active Electrons,
        Active Orbitals, Mapper, Ansatz, Optimizer, Initial Point,
        Noise Mode, Final Energy (Ha), Reference Energy (Ha),
        Error (Ha), Error (mHa), Runtime (s), Iterations, Parameters,
        Circuit Depth, CNOT Count, Alpha Electrons, Beta Electrons,
        Spatial Orbitals, Num Qubits.
    """
    n_alpha, n_beta = problem_metrics.get("num_particles", (None, None))

    data: dict[str, list[Any]] = {
        "Molecule":             [experiment.get("molecule",          "Unknown")],
        "Basis":                [experiment.get("basis",             "Unknown")],
        "Freeze Core":          [experiment.get("freeze_core",       False)],
        "Active Space":         [experiment.get("use_active_space",  False)],
        "Active Electrons":     [experiment.get("active_electrons",  "N/A")],
        "Active Orbitals":      [experiment.get("active_orbitals",   "N/A")],
        "Mapper":               [experiment.get("mapper",            "Unknown")],
        "Ansatz":               [experiment.get("ansatz",            "Unknown")],
        "Optimizer":            [experiment.get("optimizer",         "Unknown")],
        "Initial Point":        [experiment.get("initial_point",     "Unknown")],
        "Noise Mode":           ["Noisy" if experiment.get("noise", False) else "Noiseless"],
        "Final Energy (Ha)":    [round(float(vqe_energy),        8)],
        "Reference Energy (Ha)":[round(float(reference_energy),  8)],
        "Error (Ha)":           [energy_errors.get("hartree_error",   None)],
        "Error (mHa)":          [round(energy_errors.get("millihart_error", 0.0), 6)],
        "Runtime (s)":          [round(float(runtime), 3)],
        "Iterations":           [int(num_evaluations)],
        "Parameters":           [circuit_metrics.get("num_parameters", "N/A")],
        "Circuit Depth":        [circuit_metrics.get("depth",          "N/A")],
        "CNOT Count":           [circuit_metrics.get("cnot_count",     "N/A")],
        "Alpha Electrons":      [n_alpha],
        "Beta Electrons":       [n_beta],
        "Spatial Orbitals":     [problem_metrics.get("num_spatial_orbitals", None)],
        "Num Qubits":           [problem_metrics.get("num_qubits",           None)],
    }

    return pd.DataFrame(data)
