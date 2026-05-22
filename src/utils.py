# =============================================================
# utils.py
# Shared data utilities for the Mg-Ca-Zn surrogate project.
# No dependencies on other project modules.
# =============================================================

import numpy as np

# Output indices
DIFFUSIVITY_IDX = [3, 4, 5]
MU_IDX          = [0, 1, 2]
OUTPUT_NAMES    = [
    "mu_Ca", "mu_Mg", "mu_Zn",
    "Dv_Ca", "Dv_Mg", "Dv_Zn",
    "MolarVol",
]

ACCURACY_TARGET = 95.0
RAND_SEED       = 42


def transform_outputs(y: np.ndarray) -> np.ndarray:
    """
    Apply log10 transformation to diffusivity outputs (indices 3-5).
    All other outputs are left unchanged.
    """
    y_t = y.copy()
    y_t[:, DIFFUSIVITY_IDX] = np.log10(np.abs(y_t[:, DIFFUSIVITY_IDX]))
    return y_t


def inverse_transform_outputs(y_t: np.ndarray) -> np.ndarray:
    """
    Invert log10 transformation on diffusivity outputs (indices 3-5).
    All other outputs are left unchanged.
    """
    y = y_t.copy()
    y[:, DIFFUSIVITY_IDX] = 10 ** y_t[:, DIFFUSIVITY_IDX]
    return y


def engineer_features(X: np.ndarray) -> np.ndarray:
    """
    Expand 3 raw inputs (T, X_Ca, X_Zn) to 12 engineered features:
        T, X_Ca, X_Zn,          -- raw inputs
        1/T,                     -- Arrhenius linearisation
        X_Ca * X_Zn,            -- composition interaction
        X_Ca / X_Zn,            -- composition ratio
        X_Zn / T,               -- Arrhenius cross-term
        X_Ca / T,               -- Arrhenius cross-term
        X_Zn^2,                 -- quadratic composition
        X_Zn^2 / T,             -- composition-dependent activation energy
        ln(X_Ca),               -- Henry's law divergence
        ln(X_Zn),               -- Henry's law divergence
    """
    T    = X[:, 0:1]
    X_Ca = X[:, 1:2]
    X_Zn = X[:, 2:3]
    inv_T    = 1.0 / T
    log_X_Ca = np.log(np.clip(X_Ca, 1e-8, None))
    log_X_Zn = np.log(np.clip(X_Zn, 1e-8, None))
    return np.hstack([
        T, X_Ca, X_Zn,
        inv_T,
        X_Ca * X_Zn,
        X_Ca / np.clip(X_Zn, 1e-8, None),
        X_Zn * inv_T,
        X_Ca * inv_T,
        X_Zn ** 2,
        (X_Zn ** 2) * inv_T,
        log_X_Ca,
        log_X_Zn,
    ])


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute per-output accuracy as 100 - MAPE (%) in physical space.
    Diffusivity outputs must already be inverse-transformed before calling.

    Returns
    -------
    acc : np.ndarray, shape (n_outputs,)
        Accuracy percentage for each output.
    """
    return np.array([
        100.0 - np.mean(
            np.abs((y_true[:, i] - y_pred[:, i]) / np.abs(y_true[:, i]))
        ) * 100.0
        for i in range(y_true.shape[1])
    ])