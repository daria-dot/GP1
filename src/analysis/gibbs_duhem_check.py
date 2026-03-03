"""
gibbs_duhem_check.py

Verifies thermodynamic consistency of the chemical potential data
using the Gibbs-Duhem relation at constant T and P:

    X_Ca * dμ_Ca + X_Mg * dμ_Mg + X_Zn * dμ_Zn = 0

A large residual indicates either:
  - Numerical noise in the CALPHAD calculations
  - A surrogate model that has learned physically inconsistent outputs

Run directly:
    python src/analysis/gibbs_duhem_check.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
FIGURES_DIR   = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def check_gibbs_duhem(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Gibbs-Duhem residuals across adjacent composition points
    at each fixed temperature level.

    The residual is:
        GD = X_Ca_avg * Δμ_Ca + X_Mg_avg * Δμ_Mg + X_Zn_avg * Δμ_Zn

    where averages and differences are taken between neighbouring points
    sorted by X_Zn at fixed T and X_Ca.

    Returns
    -------
    pd.DataFrame of residuals with columns:
        T, X_Ca, X_Zn_from, X_Zn_to, GD_residual (J/mol)
    """
    print("=" * 70)
    print("GIBBS-DUHEM CONSISTENCY CHECK")
    print("=" * 70)
    print("  Relation: X_Ca·Δμ_Ca + X_Mg·Δμ_Mg + X_Zn·Δμ_Zn ≈ 0  (const T, P)")
    print("  Method: finite differences between adjacent X_Zn points\n")

    records = []
    T_levels = sorted(data['T'].unique())

    for T_val in T_levels:
        T_subset = data[data['T'] == T_val]
        Ca_levels = sorted(T_subset['X_Ca'].unique())

        for X_Ca_val in Ca_levels:
            subset = (T_subset[T_subset['X_Ca'] == X_Ca_val]
                      .sort_values('X_Zn')
                      .reset_index(drop=True))

            if len(subset) < 2:
                continue

            for i in range(len(subset) - 1):
                r1 = subset.iloc[i]
                r2 = subset.iloc[i + 1]

                X_avg = np.array([
                    (r1['X_Ca'] + r2['X_Ca']) / 2,
                    (r1['X_Mg'] + r2['X_Mg']) / 2,
                    (r1['X_Zn'] + r2['X_Zn']) / 2,
                ])
                d_mu = np.array([
                    r2['mu_Ca'] - r1['mu_Ca'],
                    r2['mu_Mg'] - r1['mu_Mg'],
                    r2['mu_Zn'] - r1['mu_Zn'],
                ])
                gd = float(np.dot(X_avg, d_mu))

                records.append({
                    'T':          T_val,
                    'X_Ca':       X_Ca_val,
                    'X_Zn_from':  r1['X_Zn'],
                    'X_Zn_to':    r2['X_Zn'],
                    'GD_residual': gd
                })

    results = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    abs_res = results['GD_residual'].abs()
    print(f"  Total pairs checked:     {len(results)}")
    print(f"  Mean |residual|:         {abs_res.mean():.4e} J/mol")
    print(f"  Max  |residual|:         {abs_res.max():.4e} J/mol")
    print(f"  Std  |residual|:         {abs_res.std():.4e} J/mol")
    print(f"  Pairs with |GD| < 1:     {(abs_res < 1).sum()} / {len(results)}")
    print(f"  Pairs with |GD| < 10:    {(abs_res < 10).sum()} / {len(results)}")
    print(f"  Pairs with |GD| > 100:   {(abs_res > 100).sum()} / {len(results)}")

    if abs_res.mean() < 1:
        print("\n  => EXCELLENT: Data is highly thermodynamically consistent")
    elif abs_res.mean() < 10:
        print("\n  => GOOD: Minor numerical noise, acceptable for surrogate training")
    elif abs_res.mean() < 100:
        print("\n  => WARNING: Non-trivial Gibbs-Duhem residuals — check CALPHAD data")
    else:
        print("\n  => FAIL: Large residuals — data may have physical inconsistencies")

    # Print a sample of residuals across T range
    print(f"\n  Sample residuals (first T level, first 5 pairs):")
    sample = results[results['T'] == T_levels[0]].head(5)
    for _, row in sample.iterrows():
        print(f"    T={row['T']:.0f}K, X_Ca={row['X_Ca']:.4e}, "
              f"X_Zn=[{row['X_Zn_from']:.4e}→{row['X_Zn_to']:.4e}]: "
              f"GD = {row['GD_residual']:.4e} J/mol")

    return results


def plot_gibbs_duhem(results: pd.DataFrame) -> None:
    """Save a visualisation of Gibbs-Duhem residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of residuals
    axes[0].hist(results['GD_residual'], bins=50,
                 edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1.5,
                    label='Perfect consistency (GD=0)')
    axes[0].set_xlabel('Gibbs-Duhem Residual (J/mol)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of GD Residuals', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals by temperature
    T_levels = sorted(results['T'].unique())
    means = [results[results['T'] == t]['GD_residual'].abs().mean()
             for t in T_levels]
    axes[1].plot(T_levels, means, 'o-', color='coral', linewidth=2)
    axes[1].axhline(y=1,  color='green', linestyle='--', alpha=0.6, label='1 J/mol')
    axes[1].axhline(y=10, color='orange', linestyle='--', alpha=0.6, label='10 J/mol')
    axes[1].set_xlabel('Temperature (K)', fontsize=12)
    axes[1].set_ylabel('Mean |GD Residual| (J/mol)', fontsize=12)
    axes[1].set_title('Mean GD Residual by Temperature', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / 'gibbs_duhem_check.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[gibbs_duhem] Saved {path}")


if __name__ == '__main__':
    combined_path = PROCESSED_DIR / 'combined_dataset.csv'
    if not combined_path.exists():
        raise FileNotFoundError(
            "combined_dataset.csv not found. "
            "Run data_preprocessor.py first."
        )
    data = pd.read_csv(combined_path)
    results = check_gibbs_duhem(data)
    plot_gibbs_duhem(results)