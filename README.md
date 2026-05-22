# Mg-Ca-Zn Diffusion Surrogate
### ELE469 Industry Training Programme

A data-efficient machine learning surrogate that replaces CALPHAD-DICTRA thermodynamic calculations for homogenisation simulation of Mg-Ca-Zn ternary alloys. The final model (Ensemble v3) achieves ≥95% accuracy on all seven outputs using **250 adaptively selected CALPHAD simulations** — a 99.7% reduction from the 75,000-point full dataset.

---

## Repository structure

```
GP1/
├── data/
│   ├── input_data.txt       # CALPHAD inputs:  T, X_Ca, X_Zn (75,000 points)
│   ├── output_data.txt      # CALPHAD outputs: μ_Ca, μ_Mg, μ_Zn, Dv_Ca, Dv_Mg, Dv_Zn, Vm
│   └── data.xlsx            # In718 dataset (small data + 718 composition data sheets)
├── src/
│   ├── utils.py             # Shared data utilities and constants
│   ├── model.py             # Ensemble v3: ExtraTrees + GP + PINN + KRR
│   ├── adaptive_sampling.py # Output-weighted adaptive sampling loop
│   ├── main.py              # Scarcity experiment: Adaptive vs LHS vs Sobol
│   ├── pd_convergence.py    # Partial dependence convergence study (250/500/1000)
│   └── in718.py             # Generalisation experiment: Inconel 718
├── outputs/                 # Generated automatically on first run
│   ├── figures/
│   ├── models/
│   └── *.csv
├── requirements.txt
└── README.md
```

---

## Architecture

**Ensemble v3** stacks four complementary base learners:

| Base learner | Role | Features |
|---|---|---|
| ExtraTrees | Data-efficient nonlinear regression | 12 engineered |
| GP (Matérn ν=1.5) | Calibrated uncertainty for adaptive sampling | 12 engineered |
| PINN (3×128, tanh) | Gibbs–Duhem thermodynamic constraint | 3 raw inputs |
| KRR (poly degree-3) | Composition-dependent Arrhenius structure | 12 engineered |

**Meta-learner:** multi-task Ridge for chemical potentials (μ_Ca, μ_Mg, μ_Zn); independent Ridge for each kinetic output and molar volume.

**12 engineered features:** T, X_Ca, X_Zn, 1/T, X_Ca·X_Zn, X_Ca/X_Zn, X_Zn/T, X_Ca/T, X_Zn², X_Zn²/T, ln(X_Ca), ln(X_Zn)

---

## Setup

```bash
git clone https://github.com/daria-dot/GP1.git
cd GP1
pip install -r requirements.txt
```

Python 3.10+ required. GPU optional — all experiments run on CPU.

---

## Running the experiments

All scripts are run from inside `src/`:

```bash
cd src
```

**Scarcity experiment** (Adaptive vs LHS vs Sobol, all models):
```bash
python main.py
```

**Partial dependence convergence study** (250 / 500 / 1000 samples):
```bash
python pd_convergence.py
```

**In718 generalisation experiment:**
```bash
python in718.py
```

---

## Outputs

| Script | Figures | CSVs |
|---|---|---|
| `main.py` | `outputs/figures/scarcity_comparison.png` | `scarcity_summary.csv` |
| `pd_convergence.py` | `outputs/figures/convergence/pd_convergence_*.png` | `convergence_accuracy_summary.csv` |
| `in718.py` | `outputs/figures/in718/in718_*.png` | `in718_accuracy.csv`, `in718_history.csv` |

---

## Key results

| Model | LHS | Sobol | Adaptive |
|---|---|---|---|
| Ensemble v1 | N/C | N/C | N/C |
| DNN (350k params) | 30,000 | 30,000 | 18,000 |
| PINN standalone | 3,500 | 4,000 | 3,800 |
| Ensemble v2 | 2,500 | 3,000 | 1,600 |
| **Ensemble v3** | **300** | **300** | **250** |

N/C = did not converge to ≥95% on all 7 outputs under any sample count.

---

## Module dependency

```
utils.py
    ↑
model.py
    ↑
adaptive_sampling.py
    ↑
main.py / pd_convergence.py / in718.py
```

---

## Authors

Awon · Youssef · Daria — ELE469, University of Sheffield
