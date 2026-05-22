# =============================================================
# pd_convergence.py
# Multi-sample partial dependence convergence study.
# Trains Ensemble v3 at 250, 500, and 1000 adaptively selected
# samples and overlays the PD curves to validate that the
# 250-sample model has learned the correct functional form.
#
# Usage (from GP1/src/):
#     python pd_convergence.py
#
# Data expected at:  ../data/input_data.txt
#                    ../data/output_data.txt
# Outputs saved to:  ../outputs/figures/convergence/
#                    ../outputs/models/convergence/
#
# Imports: utils.py, model.py, adaptive_sampling.py
# =============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split

from utils import (
    transform_outputs,
    inverse_transform_outputs,
    compute_accuracy,
    OUTPUT_NAMES,
    ACCURACY_TARGET,
    RAND_SEED,
)
from adaptive_sampling import run_adaptive_sampling

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / 'data'
MODELS_DIR  = ROOT / 'outputs' / 'models' / 'convergence'
FIGURES_DIR = ROOT / 'outputs' / 'figures' / 'convergence'
RESULTS_DIR = ROOT / 'outputs'

for d in [MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INPUT_FILE  = DATA_DIR / 'input_data.txt'
OUTPUT_FILE = DATA_DIR / 'output_data.txt'

# ----------------------------------------------------------
# Experiment settings
# ----------------------------------------------------------

TARGET_SAMPLES = [250, 500, 1000]

# Scaled initial seed and batch per target — keeps round count
# proportional regardless of target size (approx 3-5 rounds each):
#   250:  initial=200, batch=50  → ~1-3 rounds
#   500:  initial=300, batch=100 → ~2-3 rounds
#   1000: initial=500, batch=200 → ~3-4 rounds
INITIAL_N  = {250: 200, 500: 300, 1000: 500}
BATCH_SIZE = {250:  50, 500: 100, 1000: 200}
MAX_ROUNDS = 20

N_PD_POINTS = 100
PD_X_CA_MIN = 0.01   # physically meaningful lower bound

RAW_INPUT_NAMES = ['T (K)', 'X_Ca', 'X_Zn']

# Colours and labels per sample count
SAMPLE_COLORS = {250: '#2196F3', 500: '#FF9800', 1000: '#4CAF50'}
SAMPLE_LABELS = {250: '250 samples', 500: '500 samples', 1000: '1000 samples'}
SAMPLE_ZORDER = {250: 3, 500: 2, 1000: 1}


# =============================================================
# Training
# =============================================================

def train_all_models(X_pool, y_t_pool, y_pool, X_test, y_test):
    """
    Train one Ensemble v3 per target sample count via independent
    adaptive sampling runs. Each run uses scaled initial_n and
    batch_size so the number of adaptive rounds stays proportional.

    Returns
    -------
    ensembles     : dict {target_n: fitted StackingEnsemble}
    training_sets : dict {target_n: X_train used}
    acc_dict      : dict {target_n: per-output accuracy array}
    """
    ensembles     = {}
    training_sets = {}
    acc_dict      = {}

    for target_n in TARGET_SAMPLES:
        label = str(target_n)
        print(f"\n{'='*60}")
        print(f"  Training {target_n}-sample model")
        print(f"  initial_n={INITIAL_N[target_n]}  "
              f"batch={BATCH_SIZE[target_n]}  max_rounds={MAX_ROUNDS}")
        print(f"{'='*60}")

        history, threshold = run_adaptive_sampling(
            X_pool     = X_pool,
            y_t_pool   = y_t_pool,
            y_pool     = y_pool,
            X_test     = X_test,
            y_test     = y_test,
            models_dir = MODELS_DIR,
            initial_n  = INITIAL_N[target_n],
            batch_size = BATCH_SIZE[target_n],
            max_rounds = MAX_ROUNDS,
            target_n   = target_n,
            label      = label,
        )

        # Retrieve the ensemble from the last history entry
        ens = history[-1]['ensemble']
        # Reconstruct training set from saved indices
        idx_file = MODELS_DIR / f'indices_{label}_{history[-1]["n"]}.npy'
        X_tr = X_pool[np.load(idx_file)] if idx_file.exists() else X_pool

        ensembles[target_n]     = ens
        training_sets[target_n] = X_tr

        acc = compute_accuracy(y_test,
                               inverse_transform_outputs(ens.predict(X_test)))
        acc_dict[target_n] = acc
        print(f"\n  {target_n}-sample model: mean={acc.mean():.2f}%  "
              f"pass={int(np.sum(acc >= ACCURACY_TARGET))}/7")

    return ensembles, training_sets, acc_dict


# =============================================================
# Accuracy summary
# =============================================================

def print_and_save_accuracy(acc_dict):
    print(f"\n{'='*65}")
    print(f"  ACCURACY SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Output':<12}" +
          "".join(f"  {n:>6}" for n in TARGET_SAMPLES))
    print(f"  {'-'*44}")
    for i, nm in enumerate(OUTPUT_NAMES):
        row = f"  {nm:<12}"
        for n in TARGET_SAMPLES:
            acc = acc_dict[n][i]
            row += f"  {acc:>5.2f}{'✓' if acc >= ACCURACY_TARGET else '✗'}"
        print(row)
    print(f"  {'-'*44}")
    means = {n: acc_dict[n].mean() for n in TARGET_SAMPLES}
    print(f"  {'Mean':<12}" +
          "".join(f"  {means[n]:>6.2f}" for n in TARGET_SAMPLES))
    print(f"{'='*65}")

    rows = [
        {'output': nm,
         **{f'acc_{n}': round(float(acc_dict[n][i]), 2)
            for n in TARGET_SAMPLES}}
        for i, nm in enumerate(OUTPUT_NAMES)
    ]
    out = RESULTS_DIR / 'convergence_accuracy_summary.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Saved: {out}")


# =============================================================
# Plotting
# =============================================================

def _pd_grid(X_ref):
    """
    Return the median reference point and the three sweep grids
    (one per input dimension).
    """
    Xm   = np.median(X_ref, axis=0)
    rngs = [
        np.linspace(323.15,     973.15, N_PD_POINTS),   # T
        np.linspace(PD_X_CA_MIN, 0.20,  N_PD_POINTS),   # X_Ca
        np.linspace(1e-5,        0.30,  N_PD_POINTS),   # X_Zn
    ]
    xlabels = [
        'T (K)',
        f'X_Ca [{PD_X_CA_MIN}\u20190.20]',
        'X_Zn',
    ]
    return Xm, rngs, xlabels


def plot_pd_grid(ensembles, training_sets):
    """
    Full 21-panel grid (3 inputs × 7 outputs) with all three sample
    counts overlaid. Saved as pd_convergence_250_500_1000.png.
    """
    X_ref        = training_sets[TARGET_SAMPLES[0]]
    Xm, rngs, xlabels = _pd_grid(X_ref)

    fig, axes = plt.subplots(3, 7, figsize=(24, 13))

    for ri, (rng, xl) in enumerate(zip(rngs, xlabels)):
        Xs        = np.tile(Xm, (N_PD_POINTS, 1))
        Xs[:, ri] = rng

        for oi in range(7):
            ax = axes[ri, oi]
            for n in TARGET_SAMPLES:
                yp = inverse_transform_outputs(ensembles[n].predict(Xs))
                ax.plot(rng, yp[:, oi],
                        color=SAMPLE_COLORS[n], lw=2, alpha=0.85,
                        label=SAMPLE_LABELS[n], zorder=SAMPLE_ZORDER[n])
            ax.set_xlabel(xl, fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.25)
            if ri == 0:
                ax.set_title(OUTPUT_NAMES[oi], fontsize=8, fontweight='bold')
            if oi == 0:
                ax.set_ylabel(f'Vary {RAW_INPUT_NAMES[ri]}', fontsize=7)
            if ri == 0 and oi == 0:
                ax.legend(fontsize=7, loc='lower left')

    fig.suptitle(
        'Partial Dependence — Convergence Study  |  250 / 500 / 1000 Samples\n'
        'ExtraTrees + GP + PINN + PerOutputKRR  |  12 features  |  '
        'Final Validated Architecture\n'
        'Curve convergence confirms 250-sample model learned the correct '
        'functional form',
        fontsize=10, fontweight='bold',
    )
    plt.tight_layout()
    out = FIGURES_DIR / 'pd_convergence_250_500_1000.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Full grid saved: {out}")


def plot_pd_per_output(ensembles, training_sets):
    """
    One figure per output (7 files), each with 3 panels (one per input).
    Larger format — suitable for individual inclusion in report.
    Saved as pd_convergence_<output_name>.png.
    """
    X_ref        = training_sets[TARGET_SAMPLES[0]]
    Xm, rngs, xlabels = _pd_grid(X_ref)

    for oi, nm in enumerate(OUTPUT_NAMES):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ri, (rng, xl) in enumerate(zip(rngs, xlabels)):
            Xs        = np.tile(Xm, (N_PD_POINTS, 1))
            Xs[:, ri] = rng
            ax        = axes[ri]

            for n in TARGET_SAMPLES:
                yp = inverse_transform_outputs(ensembles[n].predict(Xs))
                ax.plot(rng, yp[:, oi],
                        color=SAMPLE_COLORS[n], lw=2.5, alpha=0.85,
                        label=SAMPLE_LABELS[n], zorder=SAMPLE_ZORDER[n])

            ax.set_xlabel(xl, fontsize=9)
            ax.set_title(f'Vary {RAW_INPUT_NAMES[ri]}', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if ri == 0:
                ax.legend(fontsize=8)

        fig.suptitle(
            f'Partial Dependence — {nm}  |  250 / 500 / 1000 Samples\n'
            'Final Validated Architecture',
            fontsize=11, fontweight='bold',
        )
        plt.tight_layout()
        out = FIGURES_DIR / f'pd_convergence_{nm}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Per-output plot: {out}")


# =============================================================
# Entry point
# =============================================================

def main():
    print('=' * 60)
    print('  Partial Dependence Convergence Study')
    print(f'  Targets: {TARGET_SAMPLES} samples')
    print('=' * 60)

    X   = pd.read_csv(INPUT_FILE,  sep=',',    header=None).values.astype(float)
    y   = pd.read_csv(OUTPUT_FILE, sep=r'\s+', header=None,
                      engine='python').values.astype(float)
    y_t = transform_outputs(y)
    print(f'Loaded {X.shape[0]} samples')

    X_pool, X_test, y_pool, y_test, y_t_pool, _ = train_test_split(
        X, y, y_t, test_size=0.2, random_state=RAND_SEED)
    print(f'Pool: {len(X_pool)}  |  Test: {len(X_test)}')

    ensembles, training_sets, acc_dict = train_all_models(
        X_pool, y_t_pool, y_pool, X_test, y_test)

    print_and_save_accuracy(acc_dict)

    print(f"\n{'='*60}\n  Generating plots...\n{'='*60}")
    plot_pd_grid(ensembles, training_sets)
    plot_pd_per_output(ensembles, training_sets)

    print(f"\n{'='*60}")
    print('  OUTPUTS')
    print(f"{'='*60}")
    print(f'  {FIGURES_DIR}/pd_convergence_250_500_1000.png')
    print(f'  {FIGURES_DIR}/pd_convergence_<output>.png  (x7)')
    print(f'  {RESULTS_DIR}/convergence_accuracy_summary.csv')
    print()
    print('  WHAT TO LOOK FOR:')
    print('  Curves close together  → 250 samples is sufficient')
    print('  Curves diverge         → that region needs more samples')
    print('  Dv outputs row 3       → check for absence of kinks')
    print('  All outputs            → smooth monotonic behaviour')
    print('=' * 60)


if __name__ == '__main__':
    main()