# =============================================================
# in718.py
# In718 Surrogate — Generalisation Experiment
# ELE469 Industry Training Programme
#
# Applies the Ensemble v3 architecture to Inconel 718,
# demonstrating cross-system generalisation without structural
# modification to the stacking framework.
#
# INPUTS  (4 raw → 12 engineered):
#   T, f_gp, f_delta, f_gpp
#
# OUTPUTS (17):
#   mu_Al, mu_Co, mu_Cr, mu_Fe, mu_Mo, mu_Nb, mu_Ni, mu_Ti  [J/mol]
#   D_Al, D_Co, D_Cr, D_Fe, D_Mo, D_Nb, D_Ni, D_Ti          [m²/s]
#   MolarVol                                                   [m³/mol]
#
# PINN physics: Henry's law soft penalty on mu_Ti and mu_Nb
# Diffusivities: log10-transformed before training, inverse-
#                transformed before accuracy evaluation
#
# Usage (from GP1/src/):
#     python in718.py
#
# Data expected at: ../data/data.xlsx
# Outputs saved to: ../outputs/figures/in718/
#                   ../outputs/models/in718/
#                   ../outputs/
#
# Imports: adaptive_sampling.py (loop), utils.py (accuracy metric)
# =============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge

from adaptive_sampling import run_adaptive_sampling
from utils import RAND_SEED, ACCURACY_TARGET

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
DATA_FILE   = ROOT / 'data'  / 'data.xlsx'
FIGURES_DIR = ROOT / 'outputs' / 'figures' / 'in718'
MODELS_DIR  = ROOT / 'outputs' / 'models'  / 'in718'
RESULTS_DIR = ROOT / 'outputs'

for d in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# In718-specific settings
# ----------------------------------------------------------

MU_NAMES  = ['mu_Al', 'mu_Co', 'mu_Cr', 'mu_Fe',
              'mu_Mo', 'mu_Nb', 'mu_Ni', 'mu_Ti']
D_NAMES   = ['D_Al',  'D_Co',  'D_Cr',  'D_Fe',
              'D_Mo',  'D_Nb',  'D_Ni',  'D_Ti']
VOL_NAMES = ['MolarVol']
OUTPUT_NAMES  = MU_NAMES + D_NAMES + VOL_NAMES   # 17 outputs

MU_IDX  = list(range(8))        # 0-7
D_IDX   = list(range(8, 16))    # 8-15
VOL_IDX = [16]

DIFFUSIVITY_IDX = D_IDX         # for log10 transform

# Per-output KRR alpha
# mu_Nb and mu_Ti diverge at depletion → higher regularisation
KRR_ALPHA = {nm: 0.01 for nm in OUTPUT_NAMES}
KRR_ALPHA['mu_Nb'] = 0.5
KRR_ALPHA['mu_Ti'] = 0.5

# PINN
HIDDEN_SIZES  = [128, 128, 128]
PINN_EPOCHS   = 200
PINN_PATIENCE = 20
PINN_LR       = 5e-4
LAMBDA_HENRY  = 0.3

N_FOLDS  = 5
N_SWEEP  = 200

# Alloy bulk composition (mole fractions) — fixed for mass balance
ALLOY_COMP = {
    'Al': 0.0097,  'Co': 0.00148, 'Cr': 0.1992,
    'Fe': 0.18859, 'Mo': 0.01741, 'Nb': 0.03388,
    'Ni': 0.53759, 'Ti': 0.01215,
}
R_GAS = 8.314   # J/mol/K

torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[OK] Device: {DEVICE}")

# Global phase composition interpolators — set in main()
_GP_TI  = _GP_NB  = _GPP_TI = _GPP_NB = None


# =============================================================
# Data utilities
# =============================================================

def transform_outputs(y: np.ndarray) -> np.ndarray:
    """log10-transform diffusivity outputs (indices 8-15)."""
    y_t = y.copy()
    y_t[:, DIFFUSIVITY_IDX] = np.log10(np.abs(y_t[:, DIFFUSIVITY_IDX]))
    return y_t


def inverse_transform_outputs(y_t: np.ndarray) -> np.ndarray:
    """Invert log10 transform on diffusivity outputs."""
    y = y_t.copy()
    y[:, DIFFUSIVITY_IDX] = 10 ** y_t[:, DIFFUSIVITY_IDX]
    return y


def engineer_features(X: np.ndarray) -> np.ndarray:
    """
    12 features from 4 raw inputs (T, f_gp, f_delta, f_gpp).

    Mirrors the Mg-Ca-Zn feature set:
      - 1/T           : Arrhenius linearisation
      - phase sums    : combined phase fraction
      - interactions  : f_gp * f_gpp, quadratics
      - combined/T    : composition-dependent activation energy
      - log(f_gp+ε)   : dilute limit (mirrors log X_Ca)
      - log(f_gpp+ε)  : dilute limit (mirrors log X_Zn)
    """
    T     = X[:, 0:1]
    f_gp  = X[:, 1:2]
    f_d   = X[:, 2:3]
    f_gpp = X[:, 3:4]
    comb  = f_gp + f_gpp
    log_gp  = np.log(np.clip(f_gp,  1e-8, None))
    log_gpp = np.log(np.clip(f_gpp, 1e-8, None))
    return np.hstack([
        T,                      # 1
        f_gp,                   # 2
        f_d,                    # 3
        f_gpp,                  # 4
        1.0 / T,                # 5
        comb,                   # 6
        f_gp * f_gpp,           # 7
        f_gp  ** 2,             # 8
        f_gpp ** 2,             # 9
        comb / T,               # 10
        log_gp,                 # 11
        log_gpp,                # 12
    ])


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-output accuracy = 100 - MAPE (%) in physical space."""
    return np.array([
        100.0 - np.mean(
            np.abs((y_true[:, i] - y_pred[:, i]) / np.abs(y_true[:, i]))
        ) * 100.0
        for i in range(y_true.shape[1])
    ])


# =============================================================
# Data loading
# =============================================================

def load_data(filepath: Path):
    """
    Load In718 dataset from 'small data' sheet of the xlsx file.
    Returns X_raw (n, 4) and y (n, 17).
    """
    print(f"Loading: {filepath}")
    df = pd.read_excel(filepath, sheet_name='small data',
                       header=None, skiprows=2)
    df.columns = [
        'idx1', 'idx2', 'T',
        'f_gp', 'f_delta', 'f_gpp',
        'X_Al', 'X_Co', 'X_Cr', 'X_Fe', 'X_Mo', 'X_Nb', 'X_Ni', 'X_Ti',
        'vf_gp', 'vf_delta', 'vf_gpp', 'MolarVol',
        'mu_Al', 'mu_Co', 'mu_Cr', 'mu_Fe', 'mu_Mo', 'mu_Nb', 'mu_Ni', 'mu_Ti',
        'D_Al', 'D_Co', 'D_Cr', 'D_Fe', 'D_Mo', 'D_Nb', 'D_Ni', 'D_Ti',
    ]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['T', 'f_gp', 'mu_Ti', 'mu_Nb'])

    X_raw = df[['T', 'f_gp', 'f_delta', 'f_gpp']].values.astype(float)
    y     = df[OUTPUT_NAMES].values.astype(float)

    print(f"  Loaded {len(X_raw)} samples  |  {y.shape[1]} outputs")
    print(f"  T:     {X_raw[:,0].min():.1f} – {X_raw[:,0].max():.1f} K")
    print(f"  f_gp:  {X_raw[:,1].min():.4f} – {X_raw[:,1].max():.4f}")
    print(f"  f_gpp: {X_raw[:,3].min():.4f} – {X_raw[:,3].max():.4f}")
    return X_raw, y


def load_phase_compositions(filepath: Path):
    """
    Load equilibrium phase compositions from '718 composition data' sheet.
    Returns linear interpolators for X_Ti and X_Nb in gamma-prime and
    gamma-double-prime phases as a function of temperature.
    Used by the PINN Henry's law penalty.
    """
    raw = pd.read_excel(filepath, sheet_name='718 composition data',
                        header=None)

    def read_phase(col_start):
        block = raw.iloc[7:, col_start:col_start+9].copy()
        block.columns = ['T', 'Al', 'Co', 'Cr', 'Fe', 'Mo', 'Nb', 'Ni', 'Ti']
        for c in block.columns:
            block[c] = pd.to_numeric(block[c], errors='coerce')
        return block.dropna().sort_values('T')

    gp  = read_phase(1)
    gpp = read_phase(21)

    kw = dict(kind='linear', bounds_error=False, fill_value='extrapolate')
    interps = (
        interp1d(gp['T'].values,  gp['Ti'].values,  **kw),
        interp1d(gp['T'].values,  gp['Nb'].values,  **kw),
        interp1d(gpp['T'].values, gpp['Ti'].values, **kw),
        interp1d(gpp['T'].values, gpp['Nb'].values, **kw),
    )
    print(f"  Phase compositions: GP {gp['T'].min():.0f}–{gp['T'].max():.0f} K"
          f"  |  GPP {gpp['T'].min():.0f}–{gpp['T'].max():.0f} K")
    return interps


def compute_matrix_composition(T, f_gp, f_gpp):
    """
    Mass balance to get matrix mole fractions of Ti and Nb:
        X_i_matrix = (X_i_alloy - X_i_gp*f_gp - X_i_gpp*f_gpp)
                     / (1 - f_gp - f_gpp)
    Clipped to [1e-10, 1] to prevent log(0) in Henry's law penalty.
    """
    f_matrix = np.clip(1.0 - f_gp - f_gpp, 1e-6, 1.0)
    X_Ti = (ALLOY_COMP['Ti']
            - _GP_TI(T)  * f_gp
            - _GPP_TI(T) * f_gpp) / f_matrix
    X_Nb = (ALLOY_COMP['Nb']
            - _GP_NB(T)  * f_gp
            - _GPP_NB(T) * f_gpp) / f_matrix
    return np.clip(X_Ti, 1e-10, 1.0), np.clip(X_Nb, 1e-10, 1.0)


# =============================================================
# PINN
# =============================================================

class In718Dataset(Dataset):
    def __init__(self, X_raw, y_raw, im, is_, om, os_):
        self.x_raw    = torch.tensor(X_raw,  dtype=torch.float32)
        self.y_raw    = torch.tensor(y_raw,  dtype=torch.float32)
        self.x_scaled = torch.tensor((X_raw - im) / is_, dtype=torch.float32)
        self.y_scaled = torch.tensor((y_raw - om) / os_, dtype=torch.float32)
        self.out_means = torch.tensor(om,  dtype=torch.float32)
        self.out_stds  = torch.tensor(os_, dtype=torch.float32)

    def __len__(self): return len(self.x_raw)

    def __getitem__(self, i):
        return (self.x_scaled[i], self.y_scaled[i],
                self.x_raw[i],   self.y_raw[i])


class In718PINN(nn.Module):
    """
    4 inputs (T, f_gp, f_delta, f_gpp) → 17 outputs.
    3 × 128 hidden layers with tanh activations.
    tanh required for Henry's law penalty (needs smooth derivatives).
    Xavier uniform initialisation throughout.
    """
    def __init__(self):
        super().__init__()
        sizes = [4] + HIDDEN_SIZES + [17]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)


def henry_law_penalty(x_raw, y_sc, out_means, out_stds):
    """
    Soft Henry's law penalty for mu_Ti (idx 7) and mu_Nb (idx 5).

    Physical basis:  mu_i = mu0_i(T) + RT * ln(X_i_matrix)
    Enforcement:     variance of (mu_i_pred - RT*ln(X_i_matrix)) across
                     the batch should be small — residual should be a
                     smooth function of T only (the mu0 reference term).

    Normalised by output standard deviation for dimensional consistency.
    """
    T_np     = x_raw[:, 0].detach().cpu().numpy()
    f_gp_np  = x_raw[:, 1].detach().cpu().numpy()
    f_gpp_np = x_raw[:, 3].detach().cpu().numpy()

    X_Ti, X_Nb = compute_matrix_composition(T_np, f_gp_np, f_gpp_np)

    T_t = x_raw[:, 0]
    RT_ln_Ti = R_GAS * T_t * torch.tensor(
        np.log(X_Ti), dtype=torch.float32, device=x_raw.device)
    RT_ln_Nb = R_GAS * T_t * torch.tensor(
        np.log(X_Nb), dtype=torch.float32, device=x_raw.device)

    mu_Ti = y_sc[:, 7] * out_stds[7] + out_means[7]
    mu_Nb = y_sc[:, 5] * out_stds[5] + out_means[5]

    res_Ti = (mu_Ti - RT_ln_Ti) / (out_stds[7] + 1e-8)
    res_Nb = (mu_Nb - RT_ln_Nb) / (out_stds[5] + 1e-8)

    return res_Ti.var() + res_Nb.var()


def train_pinn(X_train, y_t_train, ckpt_path, scaling_path):
    """
    Train the In718 PINN on log-transformed outputs.
    Identical training protocol to Mg-Ca-Zn PINN:
    Adam, lr=5e-4, cosine annealing to 1e-5, patience=20.
    """
    n     = len(X_train)
    n_val = max(10, int(n * 0.15))
    idx   = np.random.permutation(n)
    tri, vai = idx[:-n_val], idx[-n_val:]

    im  = X_train[tri].mean(0).astype(np.float32)
    is_ = X_train[tri].std(0).astype(np.float32)
    om  = y_t_train[tri].mean(0).astype(np.float32)
    os_ = y_t_train[tri].std(0).astype(np.float32)
    is_ = np.where(is_  < 1e-8, 1.0, is_)
    os_ = np.where(os_  < 1e-8, 1.0, os_)

    np.savez(scaling_path,
             in_means=im, in_stds=is_, out_means=om, out_stds=os_)

    def make_ds(ix):
        return In718Dataset(
            X_train[ix].astype(np.float32),
            y_t_train[ix].astype(np.float32),
            im, is_, om, os_,
        )

    batch = min(256, max(32, n // 4))
    trl   = DataLoader(make_ds(tri), batch_size=batch, shuffle=True)
    val   = DataLoader(make_ds(vai),
                       batch_size=min(256, max(32, n_val)), shuffle=False)

    model     = In718PINN().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=PINN_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, PINN_EPOCHS, eta_min=1e-5)
    omd = torch.tensor(om, device=DEVICE)
    osd = torch.tensor(os_, device=DEVICE)

    best, pat = float('inf'), 0
    for _ in range(1, PINN_EPOCHS + 1):
        model.train()
        for xs, ys, xr, _ in trl:
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)
            xr = xr.to(DEVICE)
            optimiser.zero_grad()
            yp  = model(xs)
            mse = nn.functional.mse_loss(yp, ys)
            hp  = henry_law_penalty(xr, yp, omd, osd)
            (mse + LAMBDA_HENRY * hp).backward()
            optimiser.step()
        scheduler.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xs, ys, _, _ in val:
                vl += nn.functional.mse_loss(
                    model(xs.to(DEVICE)), ys.to(DEVICE)).item()
        vl /= len(val)
        if vl < best:
            best, pat = vl, 0
            torch.save({'model_state_dict': model.state_dict(),
                        'hidden_sizes': HIDDEN_SIZES}, ckpt_path)
        else:
            pat += 1
            if pat >= PINN_PATIENCE:
                break


def make_pinn_wrapper(ckpt_path, scaling_path):
    """sklearn-compatible wrapper for inference."""
    class _W(BaseEstimator, RegressorMixin):
        def __init__(self, cp, sp): self.cp = cp; self.sp = sp
        def fit(self, X, y): return self
        def predict(self, X_raw):
            class _Net(nn.Module):
                def __init__(self, hs):
                    super().__init__()
                    sz = [4] + list(hs) + [17]; lrs = []
                    for i in range(len(sz) - 1):
                        lrs.append(nn.Linear(sz[i], sz[i + 1]))
                        if i < len(sz) - 2: lrs.append(nn.Tanh())
                    self.net = nn.Sequential(*lrs)
                def forward(self, x): return self.net(x)
            ckpt = torch.load(self.cp, map_location=DEVICE, weights_only=False)
            net  = _Net(ckpt.get('hidden_sizes', HIDDEN_SIZES)).to(DEVICE)
            net.load_state_dict(ckpt['model_state_dict']); net.eval()
            sp = np.load(self.sp)
            xs = (X_raw.astype(np.float32) - sp['in_means']) / sp['in_stds']
            with torch.no_grad():
                ys = net(torch.tensor(xs, dtype=torch.float32,
                                      device=DEVICE)).cpu().numpy()
            return ys * sp['out_stds'] + sp['out_means']
    return _W(ckpt_path, scaling_path)


# =============================================================
# Base learners
# =============================================================

class ExtraTreesWrapper(BaseEstimator, RegressorMixin):
    def fit(self, X_eng, y):
        self.scaler_ = StandardScaler()
        self.model_  = MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=100, max_depth=6, min_samples_leaf=5,
                bootstrap=True, random_state=RAND_SEED, n_jobs=-1))
        self.model_.fit(self.scaler_.fit_transform(X_eng), y)
        return self
    def predict(self, X_eng):
        return self.model_.predict(self.scaler_.transform(X_eng))


class GPWrapper(BaseEstimator, RegressorMixin):
    def fit(self, X_eng, y):
        self.scaler_  = StandardScaler()
        Xs = self.scaler_.fit_transform(X_eng)
        self.y_means_ = y.mean(0)
        self.y_stds_  = np.where(y.std(0) < 1e-8, 1.0, y.std(0))
        ys     = (y - self.y_means_) / self.y_stds_
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(Xs.shape[1]),
                     length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(0.1, (1e-5, 1.0))
        )
        self.models_ = [
            GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5,
                normalize_y=False).fit(Xs, ys[:, i])
            for i in range(ys.shape[1])
        ]
        return self
    def predict(self, X_eng):
        Xs = self.scaler_.transform(X_eng)
        return (np.column_stack([gp.predict(Xs) for gp in self.models_])
                * self.y_stds_ + self.y_means_)


class PerOutputKRRWrapper(BaseEstimator, RegressorMixin):
    def fit(self, X_eng, y):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X_eng)
        self.models_ = {
            nm: KernelRidge(kernel='poly', alpha=KRR_ALPHA[nm],
                            degree=3, coef0=1).fit(Xs, y[:, i])
            for i, nm in enumerate(OUTPUT_NAMES)
        }
        return self
    def predict(self, X_eng):
        Xs = self.scaler_.transform(X_eng)
        return np.column_stack(
            [self.models_[nm].predict(Xs) for nm in OUTPUT_NAMES])


# =============================================================
# In718 Stacking Ensemble
# =============================================================

class In718StackingEnsemble:
    """
    Two-level stacking ensemble for In718 (17 outputs).

    Meta-learner grouping:
      mu outputs (8)  → grouped Multi-task Ridge
      D  outputs (8)  → independent Ridge per output
      MolarVol   (1)  → independent Ridge
    """

    def __init__(self, base_learners: dict, n_folds: int = N_FOLDS):
        self.base_learners  = base_learners
        self.n_folds        = n_folds
        self.scalers_X      = {}
        self.fitted_bases   = {}
        self.meta_mu        = MultiOutputRegressor(Ridge(alpha=0.1), n_jobs=-1)
        self.scaler_mu      = StandardScaler()
        self.meta_D_        = [Ridge(alpha=0.1) for _ in range(8)]
        self.scaler_D_      = [StandardScaler() for _ in range(8)]
        self.meta_vol_      = Ridge(alpha=0.1)
        self.scaler_vol_    = StandardScaler()

    def _get_features(self, name, X, fit=False):
        if name == 'PINN': return X
        Xe = engineer_features(X)
        if fit:
            self.scalers_X[name] = StandardScaler()
            return self.scalers_X[name].fit_transform(Xe)
        return self.scalers_X[name].transform(Xe)

    def fit(self, X, y):
        n, no = y.shape
        nl    = len(self.base_learners)
        oof   = np.zeros((n, nl, no))
        kf    = KFold(n_splits=self.n_folds, shuffle=True,
                      random_state=RAND_SEED)

        for fold, (tr, va) in enumerate(kf.split(X)):
            for li, (name, lrn) in enumerate(self.base_learners.items()):
                lrn.fit(self._get_features(name, X[tr], fit=True), y[tr])
                oof[va, li, :] = lrn.predict(
                    self._get_features(name, X[va], fit=False))
            print(f"    Fold {fold+1}/{self.n_folds} done")

        # mu grouped
        oof_mu = np.zeros((n, nl * 8))
        for i, oi in enumerate(MU_IDX):
            for li in range(nl):
                oof_mu[:, li*8+i] = oof[:, li, oi]
        self.meta_mu.fit(self.scaler_mu.fit_transform(oof_mu), y[:, MU_IDX])

        # D independent
        for i, oi in enumerate(D_IDX):
            self.meta_D_[i].fit(
                self.scaler_D_[i].fit_transform(oof[:, :, oi]),
                y[:, oi])

        # MolarVol
        self.meta_vol_.fit(
            self.scaler_vol_.fit_transform(oof[:, :, VOL_IDX[0]]),
            y[:, VOL_IDX[0]])

        # Refit on full training set
        for name, lrn in self.base_learners.items():
            lrn.fit(self._get_features(name, X, fit=True), y)
            self.fitted_bases[name] = lrn
        return self

    def predict(self, X):
        nl = len(self.fitted_bases)
        n  = len(X)
        bp = np.zeros((n, nl, 17))
        for li, (name, lrn) in enumerate(self.fitted_bases.items()):
            bp[:, li, :] = lrn.predict(self._get_features(name, X))

        yp     = np.zeros((n, 17))
        oof_mu = np.zeros((n, nl * 8))
        for i, oi in enumerate(MU_IDX):
            for li in range(nl):
                oof_mu[:, li*8+i] = bp[:, li, oi]
        yp[:, MU_IDX] = self.meta_mu.predict(
            self.scaler_mu.transform(oof_mu))
        for i, oi in enumerate(D_IDX):
            yp[:, oi] = self.meta_D_[i].predict(
                self.scaler_D_[i].transform(bp[:, :, oi]))
        yp[:, VOL_IDX[0]] = self.meta_vol_.predict(
            self.scaler_vol_.transform(bp[:, :, VOL_IDX[0]]))
        return yp

    def base_predict(self, X):
        return {name: lrn.predict(self._get_features(name, X))
                for name, lrn in self.fitted_bases.items()}


def build_in718_ensemble(ckpt_path, scaling_path, n_folds=N_FOLDS):
    """Factory: fresh In718 ensemble with loaded PINN wrapper."""
    pinn = make_pinn_wrapper(ckpt_path, scaling_path)
    return In718StackingEnsemble({
        'ExtraTrees':   ExtraTreesWrapper(),
        'GP':           GPWrapper(),
        'PINN':         pinn,
        'PerOutputKRR': PerOutputKRRWrapper(),
    }, n_folds=n_folds)


# =============================================================
# =============================================================
# Plots
# =============================================================

def plot_scarcity(history: list) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ns     = [h['n']    for h in history]
    accs   = [h['acc']  for h in history]
    n_pass = [int(np.sum(h['acc'] >= ACCURACY_TARGET)) for h in history]

    nb_idx = OUTPUT_NAMES.index('mu_Nb')
    ti_idx = OUTPUT_NAMES.index('mu_Ti')

    for ax, idx, color, title in [
        (axes[0], nb_idx, '#1565C0', 'mu_Nb accuracy vs n'),
        (axes[1], ti_idx, '#2E7D32', 'mu_Ti accuracy vs n'),
    ]:
        ax.plot(ns, [a[idx] for a in accs], 'o-', color=color, lw=2)
        ax.axhline(ACCURACY_TARGET, color='red', ls='--', lw=2,
                   label='95% target')
        ax.set(xlabel='Training samples', ylabel='Accuracy (%)', title=title)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim([70, 101])

    axes[2].plot(ns, n_pass, 'o-', color='#6A1B9A', lw=2)
    axes[2].axhline(17, color='red', ls='--', lw=2, label='All 17 passing')
    axes[2].set(xlabel='Training samples',
                ylabel='Outputs passing (out of 17)',
                title='Outputs passing vs n')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 18])

    fig.suptitle(
        'In718 Surrogate — Adaptive Scarcity\n'
        'ExtraTrees + GP + PINN + PerOutputKRR  |  17 outputs  |  12 features',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = FIGURES_DIR / 'in718_scarcity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Scarcity plot saved: {out}")


def plot_parity(y_test: np.ndarray, y_pred: np.ndarray,
                acc: np.ndarray) -> None:
    """Parity plots for all 17 outputs."""
    n_cols = 6
    n_rows = int(np.ceil(17 / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4, n_rows * 4))
    axes_flat = axes.flatten()

    for i, nm in enumerate(OUTPUT_NAMES):
        ax = axes_flat[i]
        ax.scatter(y_test[:, i], y_pred[:, i],
                   alpha=0.2, s=2,
                   color='#1565C0' if i < 8 else
                         '#2E7D32' if i < 16 else '#FF9800')
        mn = min(y_test[:, i].min(), y_pred[:, i].min())
        mx = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
        ax.set_title(f'{nm}\n{acc[i]:.2f}%', fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    for j in range(17, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        'In718 — Parity Plots (all 17 outputs)\n'
        'Blue: chemical potentials  |  Green: diffusivities  |  Orange: Vm',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = FIGURES_DIR / 'in718_parity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Parity plots saved: {out}")


def plot_validation(ens, X_train: np.ndarray) -> None:
    """
    Depletion curves for mu_Nb and mu_Ti at 720°C and 650°C.
    4-panel portrait figure: 2 temperatures × 2 outputs.
    """
    f_d_med   = float(np.median(X_train[:, 2]))
    f_gp_max  = float(X_train[:, 1].max())
    f_gpp_max = float(X_train[:, 3].max())
    fracs     = np.linspace(0, 0.9, N_SWEEP)

    sweep_colors = ['#1565C0', '#2E7D32']
    sweep_labels = ["γ' sweep", "γ'' sweep"]

    fig, axes = plt.subplots(4, 1, figsize=(9, 22), constrained_layout=True)
    panel = 0

    for T_val, T_label in [(993.0, '720°C'), (923.0, '650°C')]:
        sweeps = []
        for gp_scale, gpp_scale in [(1.0, 0.0), (0.0, 1.0)]:
            X_sw = np.column_stack([
                np.full(N_SWEEP, T_val),
                fracs * f_gp_max  * gp_scale,
                np.full(N_SWEEP, f_d_med),
                fracs * f_gpp_max * gpp_scale,
            ])
            y_pred = inverse_transform_outputs(ens.predict(X_sw))
            sweeps.append((fracs * (f_gp_max * gp_scale
                                    + f_gpp_max * gpp_scale), y_pred))

        for out_name in ['mu_Nb', 'mu_Ti']:
            ax  = axes[panel]
            oi  = OUTPUT_NAMES.index(out_name)
            for si, (x_frac, yp) in enumerate(sweeps):
                ax.plot(x_frac, yp[:, oi],
                        color=sweep_colors[si], lw=2.5, alpha=0.85,
                        label=sweep_labels[si])
            ax.set_xlabel('Total phase mole fraction', fontsize=13)
            ax.set_ylabel(f'{out_name} (J/mol)', fontsize=13)
            ax.set_title(f'{out_name}  |  T = {T_val:.0f} K  ({T_label})',
                         fontsize=15, fontweight='bold')
            ax.legend(fontsize=12); ax.grid(True, alpha=0.25)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
            panel += 1

    fig.suptitle(
        'In718 Surrogate — Depletion Validation\n'
        'ExtraTrees + GP + PINN + PerOutputKRR  |  Henry\'s law constraint\n'
        'Smooth monotonic curves confirm physically valid predictions',
        fontsize=14, fontweight='bold')
    out = FIGURES_DIR / 'in718_validation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Validation portrait saved: {out}")


# =============================================================
# Entry point
# =============================================================

def main():
    global _GP_TI, _GP_NB, _GPP_TI, _GPP_NB

    print('=' * 60)
    print('  In718 Surrogate — Generalisation Experiment')
    print('  ExtraTrees + GP + PINN + PerOutputKRR')
    print('  17 outputs  |  Henry\'s law PINN constraint')
    print('=' * 60)

    # Load phase compositions for PINN Henry's law penalty
    (_GP_TI, _GP_NB, _GPP_TI, _GPP_NB) = load_phase_compositions(DATA_FILE)

    # Load data and apply log10 transform to diffusivities
    X_raw, y = load_data(DATA_FILE)
    y_t      = transform_outputs(y)

    X_pool, X_test, y_pool, y_test, y_t_pool, _ = train_test_split(
        X_raw, y, y_t, test_size=0.2, random_state=RAND_SEED)
    print(f'\nPool: {len(X_pool)}  |  Test: {len(X_test)}')

    # Run adaptive experiment via the shared loop with In718 factories
    history, threshold = run_adaptive_sampling(
        X_pool           = X_pool,
        y_t_pool         = y_t_pool,
        y_pool           = y_pool,
        X_test           = X_test,
        y_test           = y_test,
        models_dir       = MODELS_DIR,
        label            = 'in718',
        train_fn         = train_pinn,
        model_factory    = build_in718_ensemble,
        inverse_transform= inverse_transform_outputs,
        n_outputs        = len(OUTPUT_NAMES),
    )

    # Final accuracy from last round
    ens    = history[-1]['ensemble']
    y_pred = inverse_transform_outputs(ens.predict(X_test))
    acc    = compute_accuracy(y_test, y_pred)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  |  n={history[-1]['n']}")
    print(f"{'='*60}")
    print(f"  {'Output':<12} {'Accuracy':>10}  {'Pass':>6}")
    print(f"  {'-'*32}")
    for i, nm in enumerate(OUTPUT_NAMES):
        status = 'PASS ✓' if acc[i] >= ACCURACY_TARGET else 'FAIL ✗'
        print(f"  {nm:<12} {acc[i]:>9.2f}%  {status}")
    print(f"  {'Mean':<12} {acc.mean():>9.2f}%")
    print(f"  Passing: {int(np.sum(acc >= ACCURACY_TARGET))}/17")

    # Save accuracy CSV
    rows = [{'output': nm,
             'accuracy': round(float(acc[i]), 2),
             'krr_alpha': KRR_ALPHA[nm],
             'pass': bool(acc[i] >= ACCURACY_TARGET)}
            for i, nm in enumerate(OUTPUT_NAMES)]
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / 'in718_accuracy.csv', index=False)

    # Save history CSV
    hist_rows = [{'n': h['n'],
                  'n_pass': int(np.sum(h['acc'] >= ACCURACY_TARGET)),
                  **{nm: round(float(h['acc'][i]), 2)
                     for i, nm in enumerate(OUTPUT_NAMES)}}
                 for h in history]
    pd.DataFrame(hist_rows).to_csv(
        RESULTS_DIR / 'in718_history.csv', index=False)

    # Plots
    plot_scarcity(history)
    plot_parity(y_test, y_pred, acc)
    plot_validation(ens, X_pool)

    print(f"\n{'='*60}")
    print('  OUTPUT FILES')
    print(f"{'='*60}")
    print(f'  {FIGURES_DIR}/in718_scarcity.png')
    print(f'  {FIGURES_DIR}/in718_parity.png')
    print(f'  {FIGURES_DIR}/in718_validation.png')
    print(f'  {RESULTS_DIR}/in718_accuracy.csv')
    print(f'  {RESULTS_DIR}/in718_history.csv')
    if threshold:
        print(f'\n  Converged at {threshold} samples')
    print('=' * 60)


if __name__ == '__main__':
    main()