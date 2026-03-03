"""
correlation_analysis.py

Computes Pearson and Spearman correlation matrices for the 7 model outputs,
runs PCA on the transformed outputs, and saves heatmap figures.

Run directly:
    python src/analysis/correlation_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
FIGURES_DIR   = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORMED_OUT_COLS = ['mu_Ca', 'mu_Mg', 'mu_Zn',
                         'log_Dv_Ca', 'log_Dv_Mg', 'log_Dv_Zn', 'Vm']


def compute_correlations(data: pd.DataFrame) -> dict:
    """
    Compute Pearson and Spearman correlation matrices on transformed outputs.

    Parameters
    ----------
    data : pd.DataFrame
        Transformed dataset (log10 D already applied).

    Returns
    -------
    dict with 'pearson' and 'spearman' DataFrames
    """
    pearson  = data[TRANSFORMED_OUT_COLS].corr(method='pearson')
    spearman = data[TRANSFORMED_OUT_COLS].corr(method='spearman')

    print("=" * 70)
    print("PEARSON CORRELATION MATRIX (linear relationships)")
    print("=" * 70)
    print(pearson.round(4).to_string())

    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATION MATRIX (monotonic relationships)")
    print("=" * 70)
    print(spearman.round(4).to_string())

    _interpret(pearson)

    return {'pearson': pearson, 'spearman': spearman}


def _interpret(corr: pd.DataFrame) -> None:
    """Print human-readable interpretation of strongest correlations."""
    print("\n" + "=" * 70)
    print("INTERPRETATION — strongest off-diagonal correlations (|r| > 0.8)")
    print("=" * 70)
    cols = corr.columns.tolist()
    found = False
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.8:
                direction = "positively" if r > 0 else "negatively"
                print(f"  {cols[i]} vs {cols[j]}: r = {r:.4f}  ({direction} correlated)")
                found = True
    if not found:
        print("  No pairs with |r| > 0.8 — outputs are largely independent")


def plot_correlation_heatmaps(correlations: dict) -> None:
    """Save Pearson and Spearman heatmaps side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (method, corr) in zip(axes, correlations.items()):
        sns.heatmap(
            corr, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            square=True, linewidths=1,
            xticklabels=TRANSFORMED_OUT_COLS,
            yticklabels=TRANSFORMED_OUT_COLS
        )
        ax.set_title(f'{method.capitalize()} Correlation (transformed outputs)',
                     fontsize=13)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = FIGURES_DIR / 'output_correlation_heatmaps.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[correlation] Saved {path}")


def run_pca(data: pd.DataFrame) -> None:
    """Run PCA on standardised transformed outputs and print explained variance."""
    X = StandardScaler().fit_transform(data[TRANSFORMED_OUT_COLS])
    pca = PCA()
    pca.fit(X)

    evr  = pca.explained_variance_ratio_
    cumr = np.cumsum(evr)

    print("\n" + "=" * 70)
    print("PCA OF OUTPUTS")
    print("=" * 70)
    print(f"  Explained variance ratios: {evr.round(4)}")
    print(f"  Cumulative:                {cumr.round(4)}")

    n95 = int(np.argmax(cumr >= 0.95)) + 1
    print(f"\n  Components needed for 95% variance: {n95} / {len(TRANSFORMED_OUT_COLS)}")

    if n95 <= 2:
        print("  => VERY STRONG correlations — outputs are highly redundant")
        print("     A single multi-output model will be very efficient")
    elif n95 <= 4:
        print("  => MODERATE correlations — some shared structure exists")
        print("     Multi-output model recommended")
    else:
        print("  => WEAK correlations — outputs are mostly independent")
        print("     Single multi-output model still fine for ANN")

    # Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(range(1, len(evr) + 1), evr, alpha=0.7,
                color='steelblue', edgecolor='black')
    axes[0].plot(range(1, len(evr) + 1), cumr, 'ro-', linewidth=2, label='Cumulative')
    axes[0].axhline(y=0.95, color='gray', linestyle='--', alpha=0.6, label='95%')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA Scree Plot — Transformed Outputs')
    axes[0].set_xticks(range(1, len(evr) + 1))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=TRANSFORMED_OUT_COLS,
        columns=[f'PC{i+1}' for i in range(len(TRANSFORMED_OUT_COLS))]
    )
    sns.heatmap(loadings.iloc[:, :4], annot=True, fmt='.3f',
                cmap='RdBu_r', center=0, ax=axes[1], linewidths=0.5)
    axes[1].set_title('PCA Loadings (First 4 Components)')

    plt.tight_layout()
    path = FIGURES_DIR / 'pca_outputs.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[correlation] Saved {path}")


if __name__ == '__main__':
    transformed_path = PROCESSED_DIR / 'transformed_dataset.csv'
    if not transformed_path.exists():
        raise FileNotFoundError(
            "transformed_dataset.csv not found. "
            "Run data_preprocessor.py first."
        )
    data = pd.read_csv(transformed_path)
    correlations = compute_correlations(data)
    plot_correlation_heatmaps(correlations)
    run_pca(data)