"""
Estimator primitive factory for VQE simulations.

Returns a V2-compatible estimator primitive (Qiskit 2.3.0 / qiskit-algorithms
0.3.x) for use with ``qiskit_algorithms.minimum_eigensolvers.VQE``.

Noise model extensibility:
    The noisy estimator is designed so that an IBM fake-backend noise model
    can be substituted with one line. In your notebook:

        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        from qiskit_aer.noise import NoiseModel
        nm = NoiseModel.from_backend(FakeSherbrooke())
        estimator = get_estimator(noisy=True, noise_model=nm, shots=16_384)

    Without an explicit noise model, a simple two-parameter depolarising
    model is used as a demonstration baseline.
"""

from __future__ import annotations

import warnings


def get_estimator(
    noisy: bool = False,
    shots: int = 4096,
    seed: int = 123,
    noise_model=None,
):
    """Construct and return a V2-compatible estimator primitive.

    Args:
        noisy: If ``True``, return a noise-aware :class:`AerEstimatorV2`.
            If ``False`` (default), return a deterministic
            :class:`StatevectorEstimator`.
        shots: Number of measurement shots for the noisy Aer estimator.
            Ignored in noiseless mode.
        seed: Integer random seed for the Aer simulator, ensuring
            reproducible noisy results.
        noise_model: Optional pre-built :class:`qiskit_aer.noise.NoiseModel`.
            Pass a model constructed from a fake IBM backend for
            hardware-realistic simulation. If ``None`` and ``noisy=True``,
            a minimal depolarising noise model is generated automatically
            and a warning is issued.

    Returns:
        A V2-compatible estimator:

        - :class:`qiskit.primitives.StatevectorEstimator` (noiseless)
        - :class:`qiskit_aer.primitives.EstimatorV2` with noise model (noisy)

    Raises:
        ImportError: If ``noisy=True`` and ``qiskit-aer`` is not installed.

    Example:
        >>> # Noiseless
        >>> est = get_estimator()

        >>> # Noisy with default depolarising model
        >>> est = get_estimator(noisy=True, shots=8192, seed=42)

        >>> # Noisy with FakeSherbrooke calibration data
        >>> from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        >>> from qiskit_aer.noise import NoiseModel
        >>> est = get_estimator(
        ...     noisy=True,
        ...     shots=16_384,
        ...     seed=42,
        ...     noise_model=NoiseModel.from_backend(FakeSherbrooke()),
        ... )
    """
    if not noisy:
        from qiskit.primitives import StatevectorEstimator
        return StatevectorEstimator()

    # ── Noisy path ────────────────────────────────────────────────────────────
    try:
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    except ImportError as exc:
        raise ImportError(
            "qiskit-aer is required for noisy simulation. "
            "Install it with:  pip install qiskit-aer"
        ) from exc

    estimator = AerEstimatorV2()

    # Apply shot count and random seed
    _apply_options(estimator, default_shots=shots, seed_simulator=seed)

    # Resolve noise model
    if noise_model is None:
        warnings.warn(
            "No noise model supplied. Using a minimal depolarising noise "
            "model as a stand-in. For hardware-realistic simulation pass "
            "``noise_model=NoiseModel.from_backend(FakeSherbrooke())``.",
            UserWarning,
            stacklevel=2,
        )
        noise_model = _build_depolarising_noise_model()

    if noise_model is not None:
        _apply_options(estimator, noise_model=noise_model)

    return estimator


# ── Private helpers ──────────────────────────────────────────────────────────

def _apply_options(estimator, **kwargs) -> None:
    """Apply keyword options to an AerEstimatorV2 with API-version fallbacks.

    Tries ``estimator.options.<key> = value`` first; falls back to
    ``estimator.set_options(**kwargs)`` if options are not settable as
    attributes (older API).

    Args:
        estimator: An :class:`AerEstimatorV2` instance.
        **kwargs: Option name–value pairs to apply.
    """
    for key, value in kwargs.items():
        applied = False
        # Method 1: direct attribute on options object
        try:
            setattr(estimator.options, key, value)
            applied = True
        except (AttributeError, TypeError):
            pass

        if not applied:
            # Method 2: set_options callable
            try:
                estimator.set_options(**{key: value})
                applied = True
            except Exception:
                pass

        if not applied:
            warnings.warn(
                f"Could not apply estimator option '{key}={value}'. "
                "This may indicate a qiskit-aer version incompatibility.",
                UserWarning,
                stacklevel=3,
            )


def _build_depolarising_noise_model():
    """Build a minimal single- and two-qubit depolarising noise model.

    Gate error rates:
        1-qubit gates : p₁ = 0.001 (0.1 %)
        2-qubit gates : p₂ = 0.010 (1.0 %)

    This is a conservative baseline that demonstrates the effect of gate
    noise without modelling specific hardware characteristics. Replace with
    ``NoiseModel.from_backend(FakeSherbrooke())`` for device-specific noise.

    Returns:
        A :class:`qiskit_aer.noise.NoiseModel` or ``None`` if qiskit-aer
        is unavailable.
    """
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise_model = NoiseModel()

        # 1-qubit depolarising error (0.1 %)
        err_1q = depolarizing_error(0.001, 1)
        single_qubit_gates = [
            "u1", "u2", "u3",
            "rx", "ry", "rz",
            "h", "x", "y", "z", "s", "sdg", "t", "tdg",
        ]
        noise_model.add_all_qubit_quantum_error(err_1q, single_qubit_gates)

        # 2-qubit depolarising error (1.0 %)
        err_2q = depolarizing_error(0.01, 2)
        two_qubit_gates = ["cx", "ecr", "cz", "swap", "iswap"]
        noise_model.add_all_qubit_quantum_error(err_2q, two_qubit_gates)

        return noise_model

    except ImportError:
        return None
