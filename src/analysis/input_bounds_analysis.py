"""
input_bounds_analysis.py

Verifies input bounds, checks physical constraints, assesses whether
the dataset is sufficient for different surrogate methods, and saves
a bounds visualisation figure.

Run directly:
    python src/analysis/input_bounds_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
METADATA_DIR  = BASE_DIR / 'data' / 'metadata'
FIGURES_DIR   = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_COLS = ['T', 'X_Ca', 'X_Zn']
UNITS      = {'T': 'K', 'X_Ca': '-', 'X_Zn': '-'}


def analyse_bounds(data: pd.DataFrame) -> dict:
    """
    Extract and verify input bounds from the dataset.

    Returns
    -------
    dict of bounds per input column
    """
    print("=" * 70)
    print("INPUT BOUNDS ANALYSIS")
    print("=" * 70)

    bounds = {}
    print(f"\n{'Input':>8} | {'x_L':>15} | {'x_U':>15} | {'Range':>15} | {'Units':>6}")
    print("-" * 65)

    for col in INPUT_COLS:
        x_L = float(data[col].min())
        x_U = float(data[col].max())
        bounds[col] = {'x_L': x_L, 'x_U': x_U, 'units': UNITS[col]}
        print(f"{col:>8} | {x_L:>15.6e} | {x_U:>15.6e} | "
              f"{x_U - x_L:>15.6e} | {UNITS[col]:>6}")

    print("-" * 65)

    # Constraint checks
    print(f"\nPhysical constraint checks:")
    print(f"  All X_Ca >= 0:          {(data['X_Ca'] >= 0).all()}")
    print(f"  All X_Zn >= 0:          {(data['X_Zn'] >= 0).all()}")
    print(f"  All X_Ca + X_Zn <= 1:   {(data['X_Ca'] + data['X_Zn'] <= 1 + 1e-10).all()}")
    print(f"  All X_Mg >= 0:          {(data['X_Mg'] >= -1e-10).all()}")
    print(f"  All T > 0:              {(data['T'] > 0).all()}")

    # Sampling sufficiency
    n = len(data)
    d = 3
    print(f"\nSampling sufficiency (n={n}, d={d}):")
    print(f"  n^(1/d) = {n**(1/d):.1f}  (effective points per dimension)")
    print(f"  Linear RSM  needs >= {2*(d+1)}   — {'OK' if n >= 2*(d+1) else 'INSUFFICIENT'}")
    print(f"  Kriging     needs >= {10*d}   — {'OK' if n >= 10*d else 'INSUFFICIENT'}")
    print(f"  ANN         needs >= 500  — {'OK' if n >= 500 else 'INSUFFICIENT'}")

    return bounds


def plot_bounds(data: pd.DataFrame) -> None:
    """Save a 3-panel visualisation of the input domain."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Composition simplex
    ax = axes[0]
    triangle = plt.Polygon([[0, 0], [1, 0], [0, 1]],
                            fill=True, facecolor='lightblue',
                            edgecolor='black', alpha=0.2, linewidth=2)
    ax.add_patch(triangle)
    sc = ax.scatter(data['X_Ca'], data['X_Zn'], c=data['T'],
                    cmap='coolwarm', s=8, alpha=0.6, zorder=5)
    ax.set_xlabel('$X_{Ca}$', fontsize=12)
    ax.set_ylabel('$X_{Zn}$', fontsize=12)
    ax.set_title('Data Points in Composition Space', fontsize=12)
    ax.set_xlim(-0.01, 0.25)
    ax.set_ylim(-0.01, 0.35)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Temperature (K)')

    # 2. Temperature distribution
    ax = axes[1]
    unique_T = sorted(data['T'].unique())
    counts = [len(data[data['T'] == t]) for t in unique_T]
    ax.bar(range(len(unique_T)), counts,
           edgecolor='black', alpha=0.7, color='coral')
    ax.set_xticks(range(0, len(unique_T), 5))
    ax.set_xticklabels([f'{unique_T[i]:.0f}' for i in range(0, len(unique_T), 5)],
                       rotation=45, fontsize=9)
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Samples per Temperature Level', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 3. Mole fraction distributions
    ax = axes[2]
    for i, (col, color) in enumerate(zip(['X_Ca', 'X_Zn', 'X_Mg'],
                                          ['steelblue', 'coral', 'seagreen'])):
        ax.boxplot(data[col], positions=[i], widths=0.5,
                   boxprops=dict(color=color),
                   medianprops=dict(color=color, linewidth=2),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['$X_{Ca}$', '$X_{Zn}$', '$X_{Mg}$'], fontsize=12)
    ax.set_ylabel('Mole Fraction', fontsize=12)
    ax.set_title('Distribution of Mole Fractions', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / 'input_bounds_analysis.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[bounds] Saved {path}")


if __name__ == '__main__':
    combined_path = PROCESSED_DIR / 'combined_dataset.csv'
    if not combined_path.exists():
        raise FileNotFoundError(
            "combined_dataset.csv not found. "
            "Run data_preprocessor.py first."
        )
    data = pd.read_csv(combined_path)
    bounds = analyse_bounds(data)
    plot_bounds(data)