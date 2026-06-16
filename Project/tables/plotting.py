"""
Plotting utilities for VQE benchmark visualisation.

All plots use matplotlib exclusively. No Qiskit visualisation backends or
interactive widgets are used, ensuring compatibility across all notebook and
script environments.

Functions:
    plot_convergence  — Energy vs evaluation index, with optional FCI reference
                        and log-scale error subplot.
    plot_tradeoff     — Runtime vs energy error scatter for multi-run comparison.
"""

from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Use a non-interactive backend when a display is not available
try:
    matplotlib.use("Agg")  # Safe default; notebook environments override this
except Exception:
    pass

def plot_pes(r_values, vqe_energies, fci_energies=None, orca_energies=None,
             orca_r_values=None, molecule="", scan_type="", title=None,
             figsize=(12, 5), save_path=None, dpi=150):
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt

    if not r_values or not vqe_energies:
        warnings.warn("r_values or vqe_energies is empty.", UserWarning)
        return

    r_arr  = np.asarray(r_values, dtype=float)
    e_vqe  = np.asarray(vqe_energies, dtype=float)
    auto_title = title or f"PES Scan — {molecule} [{scan_type}]"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: absolute energies
    ax1 = axes[0]
    ax1.plot(r_arr, e_vqe, color="#2b6cb0", lw=1.6, marker="o", ms=5, label="VQE")
    if fci_energies is not None:
        e_fci = np.asarray(fci_energies, dtype=float)
        ax1.plot(r_arr, e_fci, color="#c53030", lw=1.4, ls="--", marker="s", ms=4, label="FCI (NumPy)")
    if orca_energies is not None:
        e_orca = np.asarray(orca_energies, dtype=float)
        r_orca = np.asarray(orca_r_values, dtype=float) if orca_r_values is not None else r_arr
        ax1.plot(r_orca, e_orca, color="#2f855a", lw=1.4, ls="-.", marker="^", ms=5, label="ORCA reference")
    ax1.set_xlabel("Bond length (Å)"); ax1.set_ylabel("Energy (Ha)")
    ax1.set_title("Potential energy surface"); ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25, ls=":")

    # Right: error vs FCI
    ax2 = axes[1]
    if fci_energies is not None:
        e_fci = np.asarray(fci_energies, dtype=float)
        err_mha = np.abs(e_vqe - e_fci) * 1000.0
        err_safe = np.where(err_mha > 0, err_mha, np.nan)
        ax2.semilogy(r_arr, err_safe, color="#c05621", lw=1.6, marker="o", ms=5, label="|VQE − FCI|")
        ax2.axhline(1.0, color="#718096", ls=":", lw=1.2, label="Chemical accuracy (1 mHa)")
        ax2.set_ylabel("Error (mHa, log scale)"); ax2.legend(fontsize=9)
        ax2.set_title("VQE error vs FCI")
    else:
        ax2.scatter(r_arr, e_vqe, color="#805ad5", s=60)
        ax2.set_ylabel("VQE Energy (Ha)"); ax2.set_title("VQE energies")
    ax2.set_xlabel("Bond length (Å)"); ax2.grid(True, alpha=0.25, ls=":")

    fig.suptitle(auto_title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

# ── Convergence plot ─────────────────────────────────────────────────────────

def plot_convergence(
    energy_history: list[float],
    reference_energy: float | None = None,
    title: str = "VQE Energy Convergence",
    figsize: tuple[int, int] = (12, 5),
    save_path: str | None = None,
    dpi: int = 150,
) -> None:
    """Plot VQE energy convergence against function evaluation index.

    Two side-by-side subplots are generated:

    - **Left** — Absolute energy (Ha) vs evaluation index, with the FCI
      reference drawn as a dashed horizontal line when provided.
    - **Right** — |E_VQE − E_FCI| on a log₁₀ scale vs evaluation index,
      with a chemical-accuracy guide line at 1 mHa. Shown only when a
      reference energy is provided; otherwise, a zoom-in of the final
      convergence region is shown.

    Args:
        energy_history: List of energy values (Ha) captured by the VQE
            callback, one per function evaluation.
        reference_energy: Reference (e.g. FCI) energy in Hartree. If
            ``None``, the error subplot is replaced by a final-region view.
        title: Figure suptitle string.
        figsize: Matplotlib figure size ``(width, height)`` in inches.
        save_path: If provided, save the figure to this path (PNG/PDF/SVG
            are supported). Directory must exist.
        dpi: Dots-per-inch for saved figures. Default: 150.

    Example:
        >>> plot_convergence(energy_history, reference_energy=-75.0129,
        ...                  title="H₂O / UCCSD / SLSQP")
    """
    if not energy_history:
        warnings.warn(
            "energy_history is empty — no convergence plot generated.",
            UserWarning,
            stacklevel=2,
        )
        return

    iterations = np.arange(1, len(energy_history) + 1)
    energies = np.asarray(energy_history, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left subplot: absolute energy ────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(
        iterations, energies,
        color="#2b6cb0", linewidth=1.4, alpha=0.9,
        label="VQE energy",
    )
    if reference_energy is not None:
        ax1.axhline(
            y=reference_energy,
            color="#c53030", linestyle="--", linewidth=1.6,
            label=f"FCI = {reference_energy:.6f} Ha",
        )
    ax1.set_xlabel("Function evaluations", fontsize=11)
    ax1.set_ylabel("Energy (Ha)", fontsize=11)
    ax1.set_title("Absolute energy", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25, linestyle=":")

    # ── Right subplot: error or zoom-in ──────────────────────────────────────
    ax2 = axes[1]
    if reference_energy is not None:
        errors_mha = np.abs(energies - reference_energy) * 1_000.0
        # Avoid log(0)
        errors_mha = np.where(errors_mha > 0, errors_mha, np.nan)
        ax2.semilogy(
            iterations, errors_mha,
            color="#c05621", linewidth=1.4, alpha=0.9,
        )
        ax2.axhline(
            y=1.0,
            color="#718096", linestyle=":", linewidth=1.2,
            label="Chemical accuracy (1 mHa)",
        )
        ax2.set_xlabel("Function evaluations", fontsize=11)
        ax2.set_ylabel("Error (mHa, log scale)", fontsize=11)
        ax2.set_title("Error vs FCI reference", fontsize=11)
        ax2.legend(fontsize=9)
    else:
        # Show the final 100 evaluations without reference
        tail = min(100, len(iterations))
        ax2.plot(
            iterations[-tail:], energies[-tail:],
            color="#805ad5", linewidth=1.4, alpha=0.9,
        )
        ax2.set_xlabel(f"Final {tail} evaluations", fontsize=11)
        ax2.set_ylabel("Energy (Ha)", fontsize=11)
        ax2.set_title("Final convergence region", fontsize=11)
    ax2.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


# ── Trade-off plot ────────────────────────────────────────────────────────────

def plot_tradeoff(
    runtime_list: list[float],
    error_list: list[float],
    labels: list[str] | None = None,
    title: str = "Runtime vs Energy Error Trade-off",
    figsize: tuple[int, int] = (8, 5),
    save_path: str | None = None,
    dpi: int = 150,
) -> None:
    """Scatter plot of runtime vs energy error for comparing VQE configurations.

    Each point represents one benchmark run. Annotates each point with its
    label. A horizontal guide line at 1 mHa marks chemical accuracy.

    When ``len(error_list) > 1``, the y-axis is set to log scale
    automatically.

    Args:
        runtime_list: List of runtimes (seconds) for each run.
        error_list: List of energy errors (mHa) for each run.
        labels: Optional list of run labels (e.g. ``"UCCSD/SLSQP"``). If
            ``None``, sequential ``"Run 1"``, ``"Run 2"``, ... labels are
            generated.
        title: Figure title string.
        figsize: Matplotlib figure size in inches.
        save_path: If provided, save the figure to this path.
        dpi: Dots-per-inch for saved figures.

    Raises:
        ValueError: If ``runtime_list`` and ``error_list`` have different
            lengths.

    Example:
        >>> plot_tradeoff(
        ...     runtime_list=[12.5, 45.3, 180.0],
        ...     error_list=[3.2, 0.8, 0.1],
        ...     labels=["UCCSD/COBYLA", "UCCSD/SLSQP", "UCCGSD/L-BFGS-B"],
        ... )
    """
    if not runtime_list or not error_list:
        warnings.warn(
            "runtime_list or error_list is empty — no trade-off plot generated.",
            UserWarning,
            stacklevel=2,
        )
        return

    if len(runtime_list) != len(error_list):
        raise ValueError(
            f"runtime_list (len={len(runtime_list)}) and "
            f"error_list (len={len(error_list)}) must have the same length."
        )

    if labels is None:
        labels = [f"Run {i + 1}" for i in range(len(runtime_list))]

    cmap = plt.cm.tab10
    n = len(runtime_list)
    colors = [cmap(i / max(n, 10)) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)

    for i, (rt, err, lbl) in enumerate(zip(runtime_list, error_list, labels)):
        ax.scatter(rt, err, s=120, color=colors[i], zorder=4, label=lbl)
        ax.annotate(
            lbl,
            xy=(rt, err),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=colors[i],
        )

    ax.axhline(
        y=1.0,
        color="#718096", linestyle="--", linewidth=1.2,
        label="Chemical accuracy (1 mHa)",
        zorder=2,
    )

    if n > 1:
        ax.set_yscale("log")

    ax.set_xlabel("Runtime (s)", fontsize=11)
    ax.set_ylabel("Energy Error (mHa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, linestyle=":")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
