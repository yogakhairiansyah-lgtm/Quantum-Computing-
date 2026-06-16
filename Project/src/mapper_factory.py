"""
Qubit mapper factory for Qiskit Nature second-quantization workflows.

Provides a unified interface for constructing Jordan-Wigner, Parity, and
Bravyi-Kitaev qubit mappers. Compatible with Qiskit Nature 0.7+ and
Qiskit 2.3.0.

Notes:
    ParityMapper supports an optional two-qubit reduction that exploits Z₂
    symmetries of the electronic Hamiltonian. Pass ``num_particles`` to enable
    it. This reduces the qubit count by 2 at the cost of restricting the
    Hilbert space to the correct particle-number sector.
"""

from __future__ import annotations

from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
    QubitMapper,
)


# ── Registry ────────────────────────────────────────────────────────────────
# Maps user-facing name strings (canonical + aliases) → internal key.
_MAPPER_REGISTRY: dict[str, str] = {
    "jordan-wigner": "jw",
    "jordanwigner":  "jw",
    "jw":            "jw",
    "parity":        "parity",
    "bravyi-kitaev": "bk",
    "bravyikitaev":  "bk",
    "bk":            "bk",
}


def get_mapper(
    name: str,
    num_particles: tuple[int, int] | None = None,
) -> QubitMapper:
    """Construct and return a Qiskit Nature qubit mapper.

    Args:
        name: Name of the qubit mapping. Case-insensitive. Supported values:

            - ``'jordan-wigner'`` or ``'jw'``  → :class:`JordanWignerMapper`
            - ``'parity'``                      → :class:`ParityMapper`
            - ``'bravyi-kitaev'`` or ``'bk'``  → :class:`BravyiKitaevMapper`

        num_particles: Optional ``(num_alpha, num_beta)`` electron tuple.
            Passed to :class:`ParityMapper` to activate two-qubit reduction.
            When set, the qubit count is reduced by 2 relative to the
            non-reduced Parity mapping. Ignored for JW and BK mappers.

    Returns:
        A fully constructed :class:`QubitMapper` instance ready for use with
        :meth:`QubitMapper.map`.

    Raises:
        ValueError: If ``name`` is not a recognized mapper identifier.

    Example:
        >>> mapper = get_mapper("jordan-wigner")
        >>> parity_2qr = get_mapper("parity", num_particles=(5, 5))
        >>> bk_mapper   = get_mapper("bk")
    """
    key = name.lower().strip().replace("-", "").replace("_", "")

    # Normalise dashes/underscores for lookup
    normalised_registry = {
        k.replace("-", "").replace("_", ""): v
        for k, v in _MAPPER_REGISTRY.items()
    }

    if key not in normalised_registry:
        supported = sorted(set(_MAPPER_REGISTRY.keys()))
        raise ValueError(
            f"Mapper '{name}' is not recognised.\n"
            f"Supported mappers : {supported}\n"
            f"  Aliases          : 'jw' -> 'jordan-wigner', "
            f"'bk' -> 'bravyi-kitaev'."
        )

    canonical = normalised_registry[key]

    if canonical == "jw":
        return JordanWignerMapper()

    if canonical == "parity":
        # num_particles=None  → no two-qubit reduction (full Hilbert space).
        # num_particles=(α,β) → two-qubit reduction enabled.
        return ParityMapper(num_particles=num_particles)

    if canonical == "bk":
        return BravyiKitaevMapper()

    raise RuntimeError(
        f"Internal error: mapper '{name}' passed validation but has no "
        "constructor. Please report this bug."
    )


def list_supported_mappers() -> list[str]:
    """Return a sorted list of all recognised mapper name strings.

    Returns:
        Sorted list of canonical names and aliases, including short forms.

    Example:
        >>> list_supported_mappers()
        ['bk', 'bravyi-kitaev', 'jw', 'jordan-wigner', 'parity']
    """
    return sorted(set(_MAPPER_REGISTRY.keys()))
