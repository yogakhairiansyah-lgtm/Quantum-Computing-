# VQE Benchmark Framework

A modular, reproducible, notebook-driven framework for benchmarking
Variational Quantum Eigensolver (VQE) trade-offs in molecular quantum chemistry.

Targets **Qiskit 2.3.0 / Qiskit Nature 0.7+ / Qiskit Algorithms 0.3+**.

---

## Overview

The framework benchmarks computational cost vs. energy accuracy for molecules
such as H₂O and CH₄ under STO-3G, across configurable combinations of:

| Dimension          | Options                                          |
|--------------------|--------------------------------------------------|
| Orbital space      | Full Space · Freeze Core · Active Space · FC+AS  |
| Qubit mapping      | Jordan-Wigner · Parity · Bravyi-Kitaev           |
| Ansatz             | UCCSD · UCCGSD · k-UpCCGSD · HEA · ADAPT-VQE    |
| Optimizer          | COBYLA · SLSQP · L-BFGS-B · SPSA                |
| Initial point      | HF (zeros) · MP2 · Zero · Random                 |
| Simulation mode    | Noiseless (StatevectorEstimator) · Noisy (Aer)   |

The **notebook is the scientific workflow**. All key steps (Freeze Core,
Active Space, Hamiltonian generation, qubit mapping, ansatz construction,
VQE execution) are explicitly visible in notebook cells. Implementation
details live in `src/`; the notebook is never reduced to a single wrapper call.

---

## Project Structure

```
project/
├── notebooks/
│   └── VQE_Benchmark.ipynb        ← Main research workflow
│
├── src/
│   ├── mapper_factory.py           ← JW / Parity / BK mapper construction
│   ├── ansatz_factory.py           ← UCCSD / UCCGSD / k-UpCCGSD / HEA
│   ├── adapt_vqe_factory.py        ← ADAPT-VQE separate builder
│   ├── optimizer_factory.py        ← COBYLA / SLSQP / L-BFGS-B / SPSA
│   ├── initial_point_factory.py    ← HF / MP2 / Zero / Random
│   ├── estimator_factory.py        ← StatevectorEstimator / AerEstimatorV2
│   ├── analysis.py                 ← Energy errors, metrics, summary tables
│   ├── plotting.py                 ← Convergence & trade-off plots
│   └── save_results.py             ← CSV persistence
│
├── configs/
│   ├── fast_config.py              ← H₂/UCCSD/COBYLA — quick tests
│   ├── balanced_config.py          ← H₂O/UCCSD/SLSQP — standard benchmark
│   └── accurate_config.py          ← H₂O/UCCGSD/L-BFGS-B + freeze core
│
├── results/                        ← CSV output directory (auto-created)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Create a Python 3.11 virtual environment
python3.11 -m venv vqe_env
source vqe_env/bin/activate        # Windows: vqe_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook
# Open notebooks/VQE_Benchmark.ipynb
```

> **Note on PySCF on Apple Silicon (macOS arm64):**
> Install PySCF via conda for native ARM support:
> `conda install -c conda-forge pyscf`

---

## Quick Start

Open `notebooks/VQE_Benchmark.ipynb` and run all cells top-to-bottom.

To switch presets, replace the `EXPERIMENT` dictionary in **Cell 1**:

```python
# Option A: Manual configuration
EXPERIMENT = { "molecule": "H2O", "ansatz": "uccsd", ... }

# Option B: Load a preset
from balanced_config import EXPERIMENT      # H2O / UCCSD / SLSQP
# from fast_config import EXPERIMENT        # H2  / UCCSD / COBYLA (fast)
# from accurate_config import EXPERIMENT    # H2O / UCCGSD / L-BFGS-B
```

---

## Benchmark Scenarios

### 1. Full Space (default)
```python
EXPERIMENT["freeze_core"]     = False
EXPERIMENT["use_active_space"] = False
```

### 2. Freeze Core only
```python
EXPERIMENT["freeze_core"]     = True
EXPERIMENT["use_active_space"] = False
```

### 3. Active Space only
```python
EXPERIMENT["freeze_core"]       = False
EXPERIMENT["use_active_space"]  = True
EXPERIMENT["active_electrons"]  = 4
EXPERIMENT["active_orbitals"]   = 4
```

### 4. Freeze Core + Active Space
```python
EXPERIMENT["freeze_core"]       = True
EXPERIMENT["use_active_space"]  = True
EXPERIMENT["active_electrons"]  = 4
EXPERIMENT["active_orbitals"]   = 4
```

### 5. ADAPT-VQE
```python
EXPERIMENT["ansatz"] = "adapt-vqe"
EXPERIMENT["optimizer"] = "slsqp"   # Gradient-based recommended
```

---

## Notebook Cell Summary

| Cell | Purpose                               |
|------|---------------------------------------|
| 1    | Experiment Setup — EXPERIMENT dict, seed |
| 2    | Molecular Problem — PySCFDriver, display system size |
| 3    | Optional Freeze Core — FreezeCoreTransformer |
| 4    | Optional Active Space — ActiveSpaceTransformer |
| 5    | Fermionic Hamiltonian — second_q_op(), display terms |
| 6    | Qubit Mapping — mapper.map(), display qubit count |
| 7    | Build VQE Components — ansatz, optimizer, estimator |
| 8    | Run VQE — construct and execute with callback |
| 9    | Analysis, Plotting, Saving — error, convergence, CSV |

---

## Extending the Framework

### Adding a new molecule
Edit the `MOLECULE_GEOMETRIES` dict in Cell 2:
```python
MOLECULE_GEOMETRIES["NH3"] = "N 0.0 0.0 0.0; H 0.94 0.0 0.0; H -0.31 0.89 0.0; H -0.31 -0.89 0.0"
```

### Enabling hardware-realistic noise
Install `qiskit-ibm-runtime` and pass a real noise model:
```python
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer.noise import NoiseModel
nm = NoiseModel.from_backend(FakeSherbrooke())
estimator = get_estimator(noisy=True, noise_model=nm, shots=16_384)
```

### Multi-run trade-off sweep
Run Cell 8 and Cell 9 in a loop with different `EXPERIMENT` configs, then
call `plot_tradeoff(runtime_list, error_list, labels)` with the aggregated
metrics.

---

## Supported Molecules (built-in geometries)

| Molecule | Basis     | Full-space electrons | Full-space orbitals | Full-space qubits* |
|----------|-----------|---------------------|---------------------|---------------------|
| H₂       | STO-3G    | 2                   | 2                   | 2                   |
| LiH      | STO-3G    | 4                   | 6                   | 10                  |
| H₂O      | STO-3G    | 10                  | 7                   | 12                  |
| CH₄      | STO-3G    | 10                  | 9                   | 16                  |
| N₂       | STO-3G    | 14                  | 10                  | 18                  |
| CO       | STO-3G    | 14                  | 10                  | 18                  |

\* Parity mapping with two-qubit reduction.

---

## References

1. Peruzzo et al., *Nature Commun.* **5**, 4213 (2014) — Original VQE.
2. Grimsley et al., *Nature Commun.* **10**, 3007 (2019) — ADAPT-VQE.
3. Lee et al., *J. Chem. Theory Comput.* **15**, 311 (2019) — k-UpCCGSD.
4. Bravyi et al., *Ann. Phys.* **298**, 210 (2002) — Bravyi-Kitaev mapping.
5. Seeley et al., *J. Chem. Phys.* **137**, 224109 (2012) — Parity mapping.
6. Qiskit Nature documentation: https://qiskit-community.github.io/qiskit-nature/
7. Qiskit Algorithms documentation: https://qiskit-community.github.io/qiskit-algorithms/

---

## License

MIT License — see `LICENSE` for details.
