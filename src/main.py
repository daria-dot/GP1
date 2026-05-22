# =============================================================
# main.py
# Scarcity experiment: Adaptive vs LHS vs Sobol sampling.
# Produces scarcity plot and CSV summary.
#
# Usage (from GP1/src/):
#     python main.py
#
# Data expected at:  ../data/input_data.txt
#                    ../data/output_data.txt
# Outputs saved to:  ../outputs/
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
from scipy.stats import qmc
from sklearn.model_selection import train_test_split

from utils import (
    transform_outputs,
    inverse_transform_outputs,
    compute_accuracy,
    OUTPUT_NAMES,
    ACCURACY_TARGET,
    RAND_SEED,
)
from model import (
    train_pinn,
    build_ensemble,
    N_FOLDS,
)
from adaptive_sampling import (
    run_adaptive_sampling,
    INITIAL_N,
    BATCH_SIZE,
)

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / 'data'
MODELS_DIR  = ROOT / 'outputs' / 'models'
RESULTS_DIR = ROOT / 'outputs'

for d in [MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INPUT_FILE  = DATA_DIR / 'input_data.txt'
OUTPUT_FILE = DATA_DIR / 'output_data.txt'

# ----------------------------------------------------------
# Passive sampling — LHS and Sobol
# ----------------------------------------------------------

def _nearest_pool_indices(design_scaled, X_pool, n_select):
    """
    For each point in design_scaled, find the nearest point in X_pool
    (Euclidean distance). Returns a list of unique pool indices of
    length n_select.
    """
    selected = []
    used     = set()
    for pt in design_scaled:
        dists = np.linalg.norm(X_pool - pt, axis=1)
        for i in used:
            dists[i] = np.inf
        idx = int(np.argmin(dists))
        selected.append(idx)
        used.add(idx)
    return selected


def lhs_indices(X_pool: np.ndarray, n: int,
                seed: int = RAND_SEED) -> list:
    """
    Select n indices from X_pool matching a Latin Hypercube design.
    Generates an LHS design in [0,1]^d then maps to pool space.
    """
    sampler = qmc.LatinHypercube(d=X_pool.shape[1], seed=seed)
    design  = sampler.random(n=n)
    X_min, X_max = X_pool.min(0), X_pool.max(0)
    design_scaled = design * (X_max - X_min) + X_min
    return _nearest_pool_indices(design_scaled, X_pool, n)


def sobol_indices(X_pool: np.ndarray, n: int,
                  seed: int = RAND_SEED) -> list:
    """
    Select n indices from X_pool matching a Sobol sequence.
    Sobol requires a power-of-2 sample count; n is rounded up and
    the first n points are used.
    """
    n_sobol = int(2 ** np.ceil(np.log2(n)))
    sampler = qmc.Sobol(d=X_pool.shape[1], scramble=True, seed=seed)
    design  = sampler.random(n=n_sobol)[:n]
    X_min, X_max = X_pool.min(0), X_pool.max(0)
    design_scaled = design * (X_max - X_min) + X_min
    return _nearest_pool_indices(design_scaled, X_pool, n)


def run_passive_sampling(
    strategy: str,
    all_indices: list,
    sample_counts: list,
    X_pool: np.ndarray,
    y_t_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list:
    """
    Evaluate the ensemble at each sample count in sample_counts using
    the pre-computed index order from an LHS or Sobol design.
    Stops early once all outputs pass ACCURACY_TARGET.

    Returns
    -------
    history : list of dicts {n, acc}
    """
    history = []
    for n in sample_counts:
        sel   = all_indices[:n]
        ia    = np.array(sel)
        X_tr  = X_pool[ia]
        y_t_tr = y_t_pool[ia]

        print(f"\n  {strategy} n={n}")
        ckpt_path    = MODELS_DIR / f'pinn_{strategy.lower()}_{n}.pt'
        scaling_path = MODELS_DIR / f'scaling_{strategy.lower()}_{n}.npz'
        train_pinn(X_tr, y_t_tr, ckpt_path, scaling_path)

        ens = build_ensemble(
            ckpt_path, scaling_path,
            n_folds=min(N_FOLDS, max(3, n // 150)),
        )
        ens.fit(X_tr, y_t_tr)

        y_pred = inverse_transform_outputs(ens.predict(X_test))
        acc    = compute_accuracy(y_test, y_pred)
        n_pass = int(np.sum(acc >= ACCURACY_TARGET))
        history.append({'n': n, 'acc': acc.copy()})
        print(f"  Mean={acc.mean():.2f}%  Pass={n_pass}/7")

        if np.all(acc >= ACCURACY_TARGET):
            print(f"  Converged at n={n}")
            break
    return history


# ----------------------------------------------------------
# Plotting
# ----------------------------------------------------------

def plot_scarcity(
    history_adaptive: list,
    history_lhs: list,
    history_sobol: list,
    adaptive_threshold: int | None,
) -> None:
    """
    Three-panel scarcity plot:
      Panel 1 — mean accuracy vs samples
      Panel 2 — worst-output accuracy vs samples
      Panel 3 — number of outputs passing vs samples
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colours = {
        'Adaptive': '#6A1B9A',
        'LHS':      '#1565C0',
        'Sobol':    '#2E7D32',
    }
    styles = {'Adaptive': '-', 'LHS': '--', 'Sobol': ':'}

    datasets = [
        ('Adaptive', history_adaptive),
        ('LHS',      history_lhs),
        ('Sobol',    history_sobol),
    ]

    titles = [
        ('Mean Accuracy (%)',          'Mean accuracy vs samples'),
        ('Worst Output Accuracy (%)',  'Worst output accuracy vs samples'),
        ('Outputs passing (out of 7)', 'Outputs passing vs samples'),
    ]
    metrics = [
        lambda h: h['acc'].mean(),
        lambda h: h['acc'].min(),
        lambda h: float(np.sum(h['acc'] >= ACCURACY_TARGET)),
    ]
    ylims = [(70, 101), (70, 101), (0, 8)]
    hlines = [ACCURACY_TARGET, ACCURACY_TARGET, 7]

    for ax, (ylabel, title), metric, ylim, hline in zip(
            axes, titles, metrics, ylims, hlines):
        for name, hist in datasets:
            ns = [h['n'] for h in hist]
            vs = [metric(h) for h in hist]
            ax.plot(ns, vs, color=colours[name], ls=styles[name],
                    lw=2, marker='o', ms=4, label=name)
        ax.axhline(hline, color='red', ls='--', lw=2,
                   label=f'{hline}% target' if hline != 7 else 'All 7 passing')
        if adaptive_threshold:
            ax.axvline(adaptive_threshold, color='#6A1B9A',
                       ls='dotted', lw=1.5, alpha=0.7,
                       label=f'Adaptive threshold ({adaptive_threshold})')
        ax.set(xlabel='Training samples', ylabel=ylabel, title=title)
        ax.set_ylim(ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Mg-Ca-Zn Surrogate — Scarcity Experiment\n'
        'ExtraTrees + GP + PINN + PerOutputKRR  |  12 features\n'
        'Adaptive (solid)  |  LHS (dashed)  |  Sobol (dotted)  '
        '|  Red = 95% target',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    out = RESULTS_DIR / 'scarcity_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Scarcity plot saved: {out}")


# ----------------------------------------------------------
# Summary CSV
# ----------------------------------------------------------

def save_summary(
    history_adaptive: list,
    history_lhs: list,
    history_sobol: list,
) -> None:
    rows = []
    for name, hist in [('Adaptive', history_adaptive),
                       ('LHS',      history_lhs),
                       ('Sobol',    history_sobol)]:
        for h in hist:
            row = {
                'strategy': name,
                'n':        h['n'],
                'mean_acc': round(float(h['acc'].mean()), 2),
                'min_acc':  round(float(h['acc'].min()),  2),
                'n_pass':   int(np.sum(h['acc'] >= ACCURACY_TARGET)),
            }
            row.update({
                nm: round(float(h['acc'][i]), 2)
                for i, nm in enumerate(OUTPUT_NAMES)
            })
            rows.append(row)
    out = RESULTS_DIR / 'scarcity_summary.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Summary saved: {out}")


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------

def main():
    print('=' * 60)
    print('  Mg-Ca-Zn Surrogate — Scarcity Experiment')
    print(f'  Start: {INITIAL_N}  |  Batch: {BATCH_SIZE}'
          f'  |  Target: {ACCURACY_TARGET}%')
    print('=' * 60)

    # Load data
    X   = pd.read_csv(INPUT_FILE,  sep=',',    header=None).values.astype(float)
    y   = pd.read_csv(OUTPUT_FILE, sep=r'\s+', header=None,
                      engine='python').values.astype(float)
    y_t = transform_outputs(y)
    print(f"Loaded {X.shape[0]} samples")

    X_pool, X_test, y_pool, y_test, y_t_pool, _ = train_test_split(
        X, y, y_t, test_size=0.2, random_state=RAND_SEED)
    print(f"Pool: {len(X_pool)}  |  Test: {len(X_test)}")

    # Sample counts for passive strategies
    MAX_PASSIVE   = 2000
    sample_counts = list(range(INITIAL_N, MAX_PASSIVE + 1, BATCH_SIZE))

    # ── Adaptive ─────────────────────────────────────────────
    print(f"\n{'=' * 60}\n  ADAPTIVE SAMPLING\n{'=' * 60}")
    history_adaptive, adaptive_threshold = run_adaptive_sampling(
        X_pool, y_t_pool, y_pool, X_test, y_test, MODELS_DIR)

    # ── LHS ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}\n  LHS SAMPLING\n{'=' * 60}")
    all_lhs = lhs_indices(X_pool, MAX_PASSIVE)
    history_lhs = run_passive_sampling(
        'LHS', all_lhs, sample_counts,
        X_pool, y_t_pool, y_pool, X_test, y_test)

    # ── Sobol ────────────────────────────────────────────────
    print(f"\n{'=' * 60}\n  SOBOL SAMPLING\n{'=' * 60}")
    all_sobol = sobol_indices(X_pool, MAX_PASSIVE)
    history_sobol = run_passive_sampling(
        'Sobol', all_sobol, sample_counts,
        X_pool, y_t_pool, y_pool, X_test, y_test)

    # ── Results ──────────────────────────────────────────────
    plot_scarcity(history_adaptive, history_lhs, history_sobol,
                  adaptive_threshold)
    save_summary(history_adaptive, history_lhs, history_sobol)

    lhs_conv   = next((h['n'] for h in history_lhs
                       if np.all(h['acc'] >= ACCURACY_TARGET)), None)
    sobol_conv = next((h['n'] for h in history_sobol
                       if np.all(h['acc'] >= ACCURACY_TARGET)), None)

    print(f"\n{'=' * 60}")
    print(f"  Adaptive threshold : {adaptive_threshold} samples")
    print(f"  LHS threshold      : {lhs_conv} samples")
    print(f"  Sobol threshold    : {sobol_conv} samples")
    print('=' * 60)


if __name__ == '__main__':
    main()