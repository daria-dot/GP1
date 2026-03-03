"""
sampling_quality.py

Assesses the quality of the input space sampling using:
  - Uniformity check (linear vs log spacing per dimension)
  - Grid completeness
  - Minimum and maximum inter-point distances
  - Discrepancy estimate (how evenly space is filled)

Run directly:
    python src/analysis/sampling_quality.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
FIGURES_DIR   = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_COLS = ['T', 'X_Ca', 'X_Zn']


def assess_sampling(data: pd.DataFrame) -> None:
    """
    Print a full sampling quality report for the input space.
    """
    print("=" * 70)
    print("SAMPLING QUALITY ASSESSMENT")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Grid completeness
    # ------------------------------------------------------------------
    n_T   = data['T'].nunique()
    n_Ca  = data['X_Ca'].nunique()
    n_Zn  = data['X_Zn'].nunique()
    expected = n_T * n_Ca * n_Zn

    print(f"\n1. GRID COMPLETENESS")
    print(f"   Unique T levels:    {n_T}")
    print(f"   Unique X_Ca levels: {n_Ca}")
    print(f"   Unique X_Zn levels: {n_Zn}")
    print(f"   Expected full grid: {expected}")
    print(f"   Actual points:      {len(data)}")
    print(f"   Completeness:       {len(data)/expected*100:.1f}%")

    # ------------------------------------------------------------------
    # 2. Spacing uniformity per dimension
    # ------------------------------------------------------------------
    print(f"\n2. SPACING UNIFORMITY PER DIMENSION")
    for col in INPUT_COLS:
        sorted_unique = np.sort(data[col].unique())
        steps = np.diff(sorted_unique)
        log_steps = np.diff(np.log10(np.abs(sorted_unique[sorted_unique > 0])))

        lin_cv  = np.std(steps) / np.mean(steps) if np.mean(steps) != 0 else 0
        log_cv  = np.std(log_steps) / np.mean(log_steps) if len(log_steps) > 0 and np.mean(log_steps) != 0 else np.inf

        spacing_type = "LINEAR" if lin_cv < 0.01 else ("LOG" if log_cv < 0.01 else "IRREGULAR")

        print(f"\n   {col}:")
        print(f"     First 3 values: {sorted_unique[:3]}")
        print(f"     Spacing CV (linear): {lin_cv:.6f}  {'✓ uniform' if lin_cv < 0.01 else ''}")
        print(f"     Spacing CV (log):    {log_cv:.6f}  {'✓ uniform' if log_cv < 0.01 else ''}")
        print(f"     => Spacing type: {spacing_type}")

    # ------------------------------------------------------------------
    # 3. Nearest-neighbour distances (on normalised inputs)
    #    Subsample for speed if dataset is large
    # ------------------------------------------------------------------
    print(f"\n3. INTER-POINT DISTANCE ANALYSIS (normalised inputs)")

    X = data[INPUT_COLS].values
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    # Subsample 1000 points for distance calculation
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_norm), size=min(1000, len(X_norm)), replace=False)
    X_sub = X_norm[idx]

    # Pairwise distances (vectorised)
    diff = X_sub[:, np.newaxis, :] - X_sub[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)  # exclude self-distance

    nn_dists = dists.min(axis=1)
    print(f"   (Using {len(X_sub)} sampled points)")
    print(f"   Min nearest-neighbour distance:  {nn_dists.min():.6f}")
    print(f"   Mean nearest-neighbour distance: {nn_dists.mean():.6f}")
    print(f"   Max nearest-neighbour distance:  {nn_dists.max():.6f}")
    print(f"   Std of NN distances:             {nn_dists.std():.6f}")

    if nn_dists.std() / nn_dists.mean() < 0.2:
        print(f"   => Very uniform spacing (low variation in NN distances)")
    elif nn_dists.std() / nn_dists.mean() < 0.5:
        print(f"   => Moderately uniform spacing")
    else:
        print(f"   => Irregular spacing — consider reviewing data generation")

    # ------------------------------------------------------------------
    # 4. Visualisation
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Spacing per dimension
    for i, col in enumerate(INPUT_COLS):
        sorted_unique = np.sort(data[col].unique())
        steps = np.diff(sorted_unique)
        axes[0].plot(range(1, len(steps) + 1), steps,
                     marker='o', markersize=4, label=col)
    axes[0].set_xlabel('Step index')
    axes[0].set_ylabel('Step size')
    axes[0].set_title('Spacing Between Consecutive Levels')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # NN distance histogram
    axes[1].hist(nn_dists, bins=30, edgecolor='black',
                 alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Nearest-neighbour distance (normalised)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('NN Distance Distribution (1000 sampled points)')
    axes[1].grid(True, alpha=0.3)

    # 2D projection of normalised inputs
    sc = axes[2].scatter(X_norm[idx, 1], X_norm[idx, 2],
                         c=X_norm[idx, 0], cmap='coolwarm',
                         s=8, alpha=0.6)
    axes[2].set_xlabel('X_Ca (normalised)')
    axes[2].set_ylabel('X_Zn (normalised)')
    axes[2].set_title('Composition Space (normalised, coloured by T)')
    plt.colorbar(sc, ax=axes[2], label='T (normalised)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / 'sampling_quality.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[sampling] Saved {path}")


if __name__ == '__main__':
    combined_path = PROCESSED_DIR / 'combined_dataset.csv'
    if not combined_path.exists():
        raise FileNotFoundError(
            "combined_dataset.csv not found. "
            "Run data_preprocessor.py first."
        )
    data = pd.read_csv(combined_path)
    assess_sampling(data)