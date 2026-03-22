"""
Stacking Ensemble Surrogate Model
Metal Additive Manufacturing - Multi-component Diffusion in Mg-Ca-Zn alloys
ELE469 Industry Training Programme

Inputs  (3): Temperature (K), Mole fraction Ca, Mole fraction Zn
Outputs (7): mu(Ca), mu(Mg), mu(Zn), Dv(Ca), Dv(Mg), Dv(Zn), Molar volume

Architecture:
  Base learners : Random Forest, Gradient Boosting, Ridge Regression
  Meta-learner  : Ridge Regression trained on out-of-fold base predictions
  Diffusivities : log10-transformed before modelling, inverse-transformed for output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

INPUT_FILE  = "GP1/data/input_data.txt"   # Temperature(K), MoleFrac(Ca), MoleFrac(Zn)
OUTPUT_FILE = "GP1/data/output_data.txt"  # mu(Ca), mu(Mg), mu(Zn), Dv(Ca), Dv(Mg), Dv(Zn), MolarVol

OUTPUT_NAMES = ["mu_Ca", "mu_Mg", "mu_Zn", "Dv_Ca", "Dv_Mg", "Dv_Zn", "MolarVol"]

# Indices of diffusivity outputs — these will be log10-transformed
DIFFUSIVITY_IDX = [3, 4, 5]

N_FOLDS   = 5    # Cross-validation folds (also used to build out-of-fold meta features)
RAND_SEED = 42

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────

def load_data(input_file, output_file):
    X = pd.read_csv(input_file, sep=',', header=None).values.astype(float)
    y = pd.read_csv(output_file, sep=r'\s+', header=None, engine='python').values.astype(float)
    print(f"Loaded {X.shape[0]} samples | {X.shape[1]} features | {y.shape[1]} outputs")
    return X, y


def transform_outputs(y, diff_idx=DIFFUSIVITY_IDX):
    """Log10-transform diffusivity columns (always positive, very small numbers)."""
    y_t = y.copy()
    y_t[:, diff_idx] = np.log10(y_t[:, diff_idx])
    return y_t


def inverse_transform_outputs(y_t, diff_idx=DIFFUSIVITY_IDX):
    """Reverse log10 transform on diffusivity columns."""
    y = y_t.copy()
    y[:, diff_idx] = 10 ** y[:, diff_idx]
    return y

# ─────────────────────────────────────────────
# 3. BASE LEARNERS
# ─────────────────────────────────────────────

def make_base_learners():
    """Return a dict of multi-output base learners."""
    rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=None,
                              random_state=RAND_SEED, n_jobs=1),
        n_jobs=1
    )
    gb = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=4, random_state=RAND_SEED),
        n_jobs=1
    )
    ridge = MultiOutputRegressor(
        Ridge(alpha=1.0),
        n_jobs=1
    )
    return {"RandomForest": rf, "GradientBoosting": gb, "Ridge": ridge}

# ─────────────────────────────────────────────
# 4. STACKING (META-LEARNING) ENSEMBLE
# ─────────────────────────────────────────────

class StackingEnsemble:
    """
    Two-level stacking ensemble.

    Level 0 : diverse base learners trained via K-Fold cross-fitting
               to generate out-of-fold (OOF) predictions for the meta-learner.
    Level 1 : meta-learner (Ridge) trained on OOF predictions.

    All base learners are also re-trained on the full training set so
    they can generate test-set meta-features.
    """

    def __init__(self, base_learners: dict, meta_learner=None, n_folds=N_FOLDS):
        self.base_learners  = base_learners
        self.meta_learner   = meta_learner or MultiOutputRegressor(Ridge(alpha=1.0))
        self.n_folds        = n_folds
        self.scalers_X      = {}   # one StandardScaler per base learner
        self.fitted_bases   = {}   # base learners fitted on full training data

    def fit(self, X, y):
        n_samples, n_outputs = y.shape
        n_base = len(self.base_learners)

        # --- Out-of-fold meta-features ---
        oof_meta = np.zeros((n_samples, n_base * n_outputs))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=RAND_SEED)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr        = y[train_idx]

            col = 0
            for name, learner in self.base_learners.items():
                scaler = StandardScaler()
                X_tr_s  = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
                learner.fit(X_tr_s, y_tr)
                preds = learner.predict(X_val_s)          # (n_val, n_outputs)
                oof_meta[val_idx, col:col + n_outputs] = preds
                col += n_outputs

            print(f"  Fold {fold_idx + 1}/{self.n_folds} complete")

        # --- Fit meta-learner on OOF predictions ---
        self.meta_scaler = StandardScaler()
        oof_meta_s = self.meta_scaler.fit_transform(oof_meta)
        self.meta_learner.fit(oof_meta_s, y)

        # --- Re-fit base learners on full training data ---
        col = 0
        for name, learner in self.base_learners.items():
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            learner.fit(X_s, y)
            self.scalers_X[name]  = scaler
            self.fitted_bases[name] = learner
            col += n_outputs

        print("Stacking ensemble fitted.")
        return self

    def predict(self, X):
        n_outputs = next(iter(self.fitted_bases.values())).predict(
            self.scalers_X[next(iter(self.scalers_X))].transform(X[:1])
        ).shape[1]

        meta_features = np.zeros((X.shape[0], len(self.fitted_bases) * n_outputs))
        col = 0
        for name, learner in self.fitted_bases.items():
            X_s = self.scalers_X[name].transform(X)
            meta_features[:, col:col + n_outputs] = learner.predict(X_s)
            col += n_outputs

        meta_features_s = self.meta_scaler.transform(meta_features)
        return self.meta_learner.predict(meta_features_s)

    def base_predict(self, X):
        """Returns dict of predictions from each base learner (for comparison)."""
        preds = {}
        for name, learner in self.fitted_bases.items():
            X_s = self.scalers_X[name].transform(X)
            preds[name] = learner.predict(X_s)
        return preds

# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

def evaluate(y_true, y_pred, label="Model"):
    """Print per-output R² and RMSE."""
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  {'Output':<14} {'R²':>10} {'RMSE':>15}")
    print(f"  {'------':<14} {'--':>10} {'----':>15}")
    for i, name in enumerate(OUTPUT_NAMES):
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        print(f"  {name:<14} {r2:>10.4f} {rmse:>15.4e}")
    overall_r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    print(f"\n  Mean R² (all outputs): {overall_r2:.4f}")
    print(f"{'─'*55}")
    return overall_r2


def plot_predicted_vs_actual(y_true, y_pred, title="Stacking Ensemble"):
    """Grid of predicted vs actual plots for all outputs."""
    n_outputs = y_true.shape[1]
    ncols = 4
    nrows = int(np.ceil(n_outputs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, OUTPUT_NAMES)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=15, color='steelblue')
        mn = min(y_true[:, i].min(), y_pred[:, i].min())
        mx = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Ideal')
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ax.set_title(f"{name}  (R²={r2:.3f})", fontsize=10)
        ax.set_xlabel("Actual", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: predicted_vs_actual.png")


def plot_feature_importance(ensemble):
    """Extract and plot feature importance from Random Forest base learner."""
    rf = ensemble.fitted_bases.get("RandomForest")
    if rf is None:
        return

    feature_names = ["Temperature (K)", "Mole frac Ca", "Mole frac Zn"]
    importances = np.array([est.feature_importances_ for est in rf.estimators_])
    # importances shape: (n_outputs, n_features)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes, OUTPUT_NAMES)):
        ax.bar(feature_names, importances[i], color=['#2196F3', '#FF9800', '#4CAF50'])
        ax.set_title(f"Feature importance\n{name}", fontsize=9)
        ax.set_ylabel("Importance", fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Random Forest Feature Importances (per output)", fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: feature_importance.png")


def cross_validate_base_learners(X, y_t):
    """5-fold CV R² for each base learner independently (on transformed outputs)."""
    print("\n── Cross-validation (5-fold, mean R²) ─────────────────")
    for name, learner in make_base_learners().items():
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        scores = cross_val_score(learner, X_s, y_t, cv=N_FOLDS,
                                 scoring='r2', n_jobs=1)
        print(f"  {name:<20}  mean={scores.mean():.4f}  std={scores.std():.4f}")

# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Stacking Ensemble — Mg-Ca-Zn Diffusion Surrogate")
    print("=" * 55)

    # Load
    X, y = load_data(INPUT_FILE, OUTPUT_FILE)

    # Transform diffusivities
    y_t = transform_outputs(y)

    # Cross-validate base learners individually (quick sanity check)
    cross_validate_base_learners(X, y_t)

    # Build and fit stacking ensemble
    print("\nFitting stacking ensemble (out-of-fold)...")
    base_learners = make_base_learners()
    ensemble = StackingEnsemble(base_learners=base_learners, n_folds=N_FOLDS)
    ensemble.fit(X, y_t)

    # Predict on full dataset (in-sample — replace with held-out test set when available)
    y_pred_t = ensemble.predict(X)

    # Inverse transform
    y_pred = inverse_transform_outputs(y_pred_t)

    # Evaluate
    evaluate(y, y_pred, label="Stacking Ensemble (in-sample)")

    # Compare individual base learners
    base_preds_t = ensemble.base_predict(X)
    for name, bp_t in base_preds_t.items():
        bp = inverse_transform_outputs(bp_t)
        evaluate(y, bp, label=f"Base: {name} (in-sample)")

    # Plots
    plot_predicted_vs_actual(y, y_pred, title="Stacking Ensemble — Predicted vs Actual")
    plot_feature_importance(ensemble)

    print("\nDone. Next steps: add a held-out test split or proper nested CV.")


if __name__ == "__main__":
    main()