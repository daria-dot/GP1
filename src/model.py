# =============================================================
# model.py
# Ensemble v3: ExtraTrees + GP + PINN + PerOutputKRR
# with a structured two-level Ridge meta-learner.
#
# Imports: utils.py
# =============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from utils import (
    engineer_features,
    OUTPUT_NAMES, MU_IDX, DIFFUSIVITY_IDX,
    RAND_SEED,
)

# ----------------------------------------------------------
# Global settings
# ----------------------------------------------------------

HIDDEN_SIZES  = [128, 128, 128]
PINN_EPOCHS   = 200
PINN_PATIENCE = 20
PINN_LR       = 5e-4
LAMBDA_GD     = 0.5
N_FOLDS       = 5

KRR_ALPHA = {
    'mu_Ca':    0.01,
    'mu_Mg':    0.01,
    'mu_Zn':    0.01,
    'Dv_Ca':    0.5,
    'Dv_Mg':    0.5,
    'Dv_Zn':    0.01,
    'MolarVol': 0.01,
}

torch.manual_seed(RAND_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================
# PINN — architecture, Gibbs–Duhem penalty, training loop
# =============================================================

class AlloyDataset(Dataset):
    """
    PyTorch dataset that stores both raw and standardised inputs/outputs.
    Raw values are needed for the Gibbs–Duhem penalty computation.
    """
    def __init__(self, X_raw, y_t_raw, in_means, in_stds, out_means, out_stds):
        self.x_raw    = torch.tensor(X_raw,   dtype=torch.float32)
        self.y_raw    = torch.tensor(y_t_raw, dtype=torch.float32)
        self.x_scaled = torch.tensor(
            (X_raw - in_means) / in_stds, dtype=torch.float32)
        self.y_scaled = torch.tensor(
            (y_t_raw - out_means) / out_stds, dtype=torch.float32)
        self.out_means = torch.tensor(out_means, dtype=torch.float32)
        self.out_stds  = torch.tensor(out_stds,  dtype=torch.float32)

    def __len__(self):
        return len(self.x_raw)

    def __getitem__(self, i):
        return (self.x_scaled[i], self.y_scaled[i],
                self.x_raw[i],   self.y_raw[i])


class PhysicsInformedMLP(nn.Module):
    """
    Fully connected network 3 -> 128 -> 128 -> 128 -> 7 with tanh
    activations. tanh is required (not ReLU) so that autograd produces
    non-zero second derivatives for the Gibbs–Duhem penalty.
    Weights initialised with Xavier uniform; biases initialised to zero.
    """
    def __init__(self):
        super().__init__()
        sizes = [3] + HIDDEN_SIZES + [7]
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

    def forward(self, x):
        return self.net(x)


def gibbs_duhem_penalty(x_scaled, y_scaled, x_raw, out_means, out_stds):
    """
    Compute the soft Gibbs–Duhem penalty:
        L_GD = mean( (X_Ca * d(mu_Ca)/dX_j
                    + X_Mg * d(mu_Mg)/dX_j
                    + X_Zn * d(mu_Zn)/dX_j)^2 )
    for j in {Ca (col 1), Zn (col 2)}.

    Gradients are computed via autograd on the scaled network output,
    then un-standardised before the identity is evaluated.
    Residual is normalised by mean output std to keep it dimensionless.

    Returns
    -------
    penalty : scalar tensor (differentiable)
    mean_residual : scalar tensor (detached, for logging)
    """
    mu = y_scaled[:, :3] * out_stds[:3] + out_means[:3]
    Ca  = x_raw[:, 1]
    Zn  = x_raw[:, 2]
    Mg  = 1.0 - Ca - Zn
    residuals = []
    for j in [1, 2]:   # composition columns: X_Ca=1, X_Zn=2
        grads = [
            torch.autograd.grad(
                mu[:, i].sum(), x_scaled,
                create_graph=True, retain_graph=True
            )[0][:, j]
            for i in range(3)
        ]
        residuals.append(Ca * grads[0] + Mg * grads[1] + Zn * grads[2])
    stacked = torch.stack(residuals, dim=1)
    penalty       = ((stacked / out_stds[:3].mean()) ** 2).mean()
    mean_residual = stacked.abs().mean().detach()
    return penalty, mean_residual


def train_pinn(X_train, y_t_train, ckpt_path, scaling_path):
    """
    Train the PINN on (X_train, y_t_train) where diffusivities are
    already log10-transformed. Saves the best checkpoint and scaling
    parameters.

    A 15% validation split (min 10 points) is used for early stopping.
    PINN is retrained from scratch each call — no warm-starting.

    Parameters
    ----------
    X_train      : np.ndarray (n, 3)   raw inputs
    y_t_train    : np.ndarray (n, 7)   log-transformed outputs
    ckpt_path    : Path                where to save model weights
    scaling_path : Path                where to save scaling .npz
    """
    n     = len(X_train)
    n_val = max(10, int(n * 0.15))
    idx   = np.random.permutation(n)
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]

    in_means  = X_train[tr_idx].mean(0).astype(np.float32)
    in_stds   = X_train[tr_idx].std(0).astype(np.float32)
    out_means = y_t_train[tr_idx].mean(0).astype(np.float32)
    out_stds  = y_t_train[tr_idx].std(0).astype(np.float32)
    in_stds   = np.where(in_stds  < 1e-8, 1.0, in_stds)
    out_stds  = np.where(out_stds < 1e-8, 1.0, out_stds)

    np.savez(scaling_path,
             in_means=in_means, in_stds=in_stds,
             out_means=out_means, out_stds=out_stds)

    def make_dataset(idx):
        return AlloyDataset(
            X_train[idx].astype(np.float32),
            y_t_train[idx].astype(np.float32),
            in_means, in_stds, out_means, out_stds,
        )

    batch = min(256, max(32, n // 4))
    train_loader = DataLoader(make_dataset(tr_idx),
                              batch_size=batch, shuffle=True)
    val_loader   = DataLoader(make_dataset(va_idx),
                              batch_size=min(256, max(32, n_val)),
                              shuffle=False)

    model    = PhysicsInformedMLP().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=PINN_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, PINN_EPOCHS, eta_min=1e-5)
    omd = torch.tensor(out_means, device=DEVICE)
    osd = torch.tensor(out_stds,  device=DEVICE)

    best_val, patience_count = float('inf'), 0
    for epoch in range(1, PINN_EPOCHS + 1):
        model.train()
        for xs, ys, xr, _ in train_loader:
            xs = xs.to(DEVICE).requires_grad_(True)
            ys = ys.to(DEVICE)
            xr = xr.to(DEVICE)
            optimiser.zero_grad()
            yp  = model(xs)
            mse = nn.functional.mse_loss(yp, ys)
            gp, _ = gibbs_duhem_penalty(xs, yp, xr, omd, osd)
            (mse + LAMBDA_GD * gp).backward()
            optimiser.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xs, ys, _, _ in val_loader:
                val_loss += nn.functional.mse_loss(
                    model(xs.to(DEVICE)), ys.to(DEVICE)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val, patience_count = val_loss, 0
            torch.save(
                {'model_state_dict': model.state_dict(),
                 'hidden_sizes':     HIDDEN_SIZES},
                ckpt_path,
            )
        else:
            patience_count += 1
            if patience_count >= PINN_PATIENCE:
                break


def make_pinn_wrapper(ckpt_path, scaling_path):
    """
    Return a sklearn-compatible wrapper that loads the saved PINN
    checkpoint and scaling parameters for inference.
    """
    class _PINNWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, cp, sp):
            self.cp = cp
            self.sp = sp

        def fit(self, X, y):
            return self   # training handled externally

        def predict(self, X_raw):
            # Rebuild architecture from checkpoint
            ckpt = torch.load(self.cp, map_location=DEVICE,
                              weights_only=False)
            hs   = ckpt.get('hidden_sizes', HIDDEN_SIZES)

            class _Net(nn.Module):
                def __init__(self, hidden_sizes):
                    super().__init__()
                    sizes = [3] + list(hidden_sizes) + [7]
                    lrs = []
                    for i in range(len(sizes) - 1):
                        lrs.append(nn.Linear(sizes[i], sizes[i + 1]))
                        if i < len(sizes) - 2:
                            lrs.append(nn.Tanh())
                    self.net = nn.Sequential(*lrs)
                def forward(self, x):
                    return self.net(x)

            net = _Net(hs).to(DEVICE)
            net.load_state_dict(ckpt['model_state_dict'])
            net.eval()

            sp = np.load(self.sp)
            xs = ((X_raw.astype(np.float32) - sp['in_means'])
                  / sp['in_stds'])
            with torch.no_grad():
                ys = net(torch.tensor(xs, dtype=torch.float32,
                                      device=DEVICE)).cpu().numpy()
            return ys * sp['out_stds'] + sp['out_means']

    return _PINNWrapper(ckpt_path, scaling_path)


# =============================================================
# Base learner wrappers
# =============================================================

class ExtraTreesWrapper(BaseEstimator, RegressorMixin):
    """
    ExtraTrees with internal StandardScaler on engineered features.
    n_estimators=100, max_depth=6, min_samples_leaf=5, bootstrap=True.
    """
    def fit(self, X_eng, y):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X_eng)
        self.model_ = MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=5,
                bootstrap=True,
                random_state=RAND_SEED,
                n_jobs=-1,
            ),
            n_jobs=1,
        )
        self.model_.fit(Xs, y)
        return self

    def predict(self, X_eng):
        return self.model_.predict(self.scaler_.transform(X_eng))


class GPWrapper(BaseEstimator, RegressorMixin):
    """
    Seven independent GPs (one per output) with anisotropic
    Matern nu=1.5 kernel + WhiteKernel noise term.
    Outputs are internally standardised; hyperparameters optimised
    via maximum likelihood with n_restarts=5.
    """
    def fit(self, X_eng, y):
        self.scaler_  = StandardScaler()
        Xs = self.scaler_.fit_transform(X_eng)
        self.y_means_ = y.mean(0)
        self.y_stds_  = np.where(y.std(0) < 1e-8, 1.0, y.std(0))
        ys = (y - self.y_means_) / self.y_stds_
        n_feat = Xs.shape[1]
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=np.ones(n_feat),
                     length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(noise_level=0.1,
                          noise_level_bounds=(1e-5, 1.0))
        )
        self.models_ = [
            GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=False,
            ).fit(Xs, ys[:, i])
            for i in range(ys.shape[1])
        ]
        return self

    def predict(self, X_eng):
        Xs = self.scaler_.transform(X_eng)
        return (
            np.column_stack([gp.predict(Xs) for gp in self.models_])
            * self.y_stds_ + self.y_means_
        )


class PerOutputKRRWrapper(BaseEstimator, RegressorMixin):
    """
    Per-output Kernel Ridge Regression with degree-3 polynomial kernel.
    Per-output regularisation: alpha=0.5 for Dv_Ca and Dv_Mg (boundary
    oscillation suppression); alpha=0.01 for all other outputs.
    """
    def fit(self, X_eng, y):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X_eng)
        self.models_ = {
            name: KernelRidge(
                kernel='poly',
                alpha=KRR_ALPHA[name],
                degree=3,
                coef0=1,
            ).fit(Xs, y[:, i])
            for i, name in enumerate(OUTPUT_NAMES)
        }
        return self

    def predict(self, X_eng):
        Xs = self.scaler_.transform(X_eng)
        return np.column_stack(
            [self.models_[name].predict(Xs) for name in OUTPUT_NAMES]
        )


# =============================================================
# Stacking Ensemble
# =============================================================

class StackingEnsemble:
    """
    Two-level stacking ensemble (Equation 11 in report).

    Base learners
    -------------
    ExtraTrees, GP, PINN, PerOutputKRR

    Meta-learner
    ------------
    Multi-task Ridge for chemical potentials (mu_Ca, mu_Mg, mu_Zn),
    trained on a 12-feature OOF matrix (4 learners x 3 outputs).

    Four independent Ridge regressors for kinetic outputs and molar
    volume (Dv_Ca, Dv_Mg, Dv_Zn, Vm), each seeing a 4-feature OOF
    vector (one prediction per base learner).

    Out-of-fold predictions are generated via N_FOLDS-fold
    cross-fitting to prevent the meta-learner from overfitting
    to base learner training errors.
    """

    def __init__(self, base_learners: dict, n_folds: int = N_FOLDS):
        self.base_learners  = base_learners
        self.n_folds        = n_folds
        self.scalers_X      = {}
        self.fitted_bases   = {}
        self.meta_mu        = MultiOutputRegressor(
                                  Ridge(alpha=0.1), n_jobs=-1)
        self.scaler_mu      = StandardScaler()
        self.meta_others_   = [Ridge(alpha=0.1) for _ in range(4)]
        self.scaler_others_ = [StandardScaler()  for _ in range(4)]

    def _get_features(self, name: str, X: np.ndarray,
                      fit: bool = False) -> np.ndarray:
        """
        Return the appropriate feature matrix for a given base learner.
        PINN receives 3 raw inputs; all others receive 12 engineered
        features scaled by a per-learner StandardScaler.
        """
        if name == 'PINN':
            return X
        Xe = engineer_features(X)
        if fit:
            self.scalers_X[name] = StandardScaler()
            return self.scalers_X[name].fit_transform(Xe)
        return self.scalers_X[name].transform(Xe)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, n_out = y.shape
        nl       = len(self.base_learners)
        oof      = np.zeros((n, nl, n_out))

        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=RAND_SEED)
        for fold, (tr, va) in enumerate(kf.split(X)):
            for li, (name, lrn) in enumerate(self.base_learners.items()):
                lrn.fit(self._get_features(name, X[tr], fit=True), y[tr])
                oof[va, li, :] = lrn.predict(
                    self._get_features(name, X[va], fit=False))
            print(f"    Fold {fold + 1}/{self.n_folds} done")

        # Fit chemical potential meta-learner
        oof_mu = np.zeros((n, nl * 3))
        for i, out_idx in enumerate(MU_IDX):
            for li in range(nl):
                oof_mu[:, li * 3 + i] = oof[:, li, out_idx]
        self.meta_mu.fit(
            self.scaler_mu.fit_transform(oof_mu), y[:, MU_IDX])

        # Fit independent meta-learners for kinetic outputs + Vm
        for i, out_idx in enumerate([3, 4, 5, 6]):
            self.meta_others_[i].fit(
                self.scaler_others_[i].fit_transform(oof[:, :, out_idx]),
                y[:, out_idx],
            )

        # Refit all base learners on the full training set
        for name, lrn in self.base_learners.items():
            lrn.fit(self._get_features(name, X, fit=True), y)
            self.fitted_bases[name] = lrn
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        nl = len(self.fitted_bases)
        n  = len(X)
        bp = np.zeros((n, nl, 7))
        for li, (name, lrn) in enumerate(self.fitted_bases.items()):
            bp[:, li, :] = lrn.predict(self._get_features(name, X))

        yp     = np.zeros((n, 7))
        oof_mu = np.zeros((n, nl * 3))
        for i, out_idx in enumerate(MU_IDX):
            for li in range(nl):
                oof_mu[:, li * 3 + i] = bp[:, li, out_idx]
        yp[:, MU_IDX] = self.meta_mu.predict(
            self.scaler_mu.transform(oof_mu))
        for i, out_idx in enumerate([3, 4, 5, 6]):
            yp[:, out_idx] = self.meta_others_[i].predict(
                self.scaler_others_[i].transform(bp[:, :, out_idx]))
        return yp

    def base_predict(self, X: np.ndarray) -> dict:
        """Return a dict of {learner_name: predictions} for uncertainty estimation."""
        return {
            name: lrn.predict(self._get_features(name, X))
            for name, lrn in self.fitted_bases.items()
        }


def build_ensemble(ckpt_path: Path, scaling_path: Path,
                   n_folds: int = N_FOLDS) -> StackingEnsemble:
    """
    Convenience factory: instantiate a fresh Ensemble v3 with all
    four base learners. The PINN wrapper is loaded from the given
    checkpoint and scaling paths.
    """
    pinn = make_pinn_wrapper(ckpt_path, scaling_path)
    base_learners = {
        'ExtraTrees':   ExtraTreesWrapper(),
        'GP':           GPWrapper(),
        'PINN':         pinn,
        'PerOutputKRR': PerOutputKRRWrapper(),
    }
    return StackingEnsemble(base_learners, n_folds=n_folds)