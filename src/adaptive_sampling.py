# =============================================================
# adaptive_sampling.py
# Output-weighted adaptive sampling loop for Ensemble v3.
#
# Imports: utils.py, model.py
# =============================================================

import numpy as np
from pathlib import Path
from typing import Callable

from utils import (
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

# ----------------------------------------------------------
# Default settings (used by main.py scarcity experiment)
# ----------------------------------------------------------

DEFAULT_INITIAL_N  = 100
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_ROUNDS = 200
DIVERSITY_WEIGHT   = 0.5


# =============================================================
# Sampling utilities
# =============================================================

def compute_output_weights(
    acc: np.ndarray,
    target: float = ACCURACY_TARGET,
) -> np.ndarray:
    """
    Compute per-output sampling weights proportional to the accuracy
    gap below the target. Outputs already passing receive zero weight.
    If all outputs are passing, weights are uniform (should not occur
    mid-loop but included for safety).

    Parameters
    ----------
    acc    : per-output accuracy array (%)
    target : accuracy threshold (default 95.0)

    Returns
    -------
    weights : np.ndarray, shape (n_outputs,), sums to 1
    """
    gaps  = np.maximum(target - acc, 0.0)
    total = gaps.sum()
    return gaps / total if total > 0 else np.ones(len(acc)) / len(acc)


def diverse_batch_select(
    uncertainty: np.ndarray,
    X: np.ndarray,
    batch_size: int,
    diversity_weight: float = DIVERSITY_WEIGHT,
) -> np.ndarray:
    """
    Greedy furthest-point batch selection balancing uncertainty and
    spatial diversity:

        score = (1 - dw) * uncertainty + dw * min_distance_to_selected

    Both uncertainty and min-distance are normalised to [0, 1] before
    combining. Already-selected points are masked out each iteration.

    Parameters
    ----------
    uncertainty      : per-candidate scalar score
    X                : candidate input coordinates (n, d)
    batch_size       : number of points to select
    diversity_weight : weight on diversity term (0 = pure uncertainty)

    Returns
    -------
    selected : np.ndarray of selected indices into X
    """
    n          = len(uncertainty)
    batch_size = min(batch_size, n)

    u_norm = ((uncertainty - uncertainty.min())
              / (uncertainty.max() - uncertainty.min() + 1e-10))
    X_norm = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-10)

    selected = []
    min_dist = np.full(n, np.inf)

    for _ in range(batch_size):
        d      = min_dist.copy()
        finite = d[np.isfinite(d)]
        if len(finite) > 0:
            d = np.where(
                np.isfinite(d),
                (d - finite.min()) / (finite.max() - finite.min() + 1e-10),
                1.0,
            )
        score           = (1 - diversity_weight) * u_norm + diversity_weight * d
        score[selected] = -np.inf
        best            = int(np.argmax(score))
        selected.append(best)
        min_dist = np.minimum(
            min_dist,
            np.linalg.norm(X_norm - X_norm[best], axis=1),
        )
    return np.array(selected)


# =============================================================
# Adaptive sampling loop
# =============================================================

def run_adaptive_sampling(
    X_pool: np.ndarray,
    y_t_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models_dir: Path,
    initial_n: int = DEFAULT_INITIAL_N,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    target_n: int | None = None,
    label: str = 'adaptive',
    train_fn: Callable | None = None,
    model_factory: Callable | None = None,
    inverse_transform: Callable | None = None,
    n_outputs: int | None = None,
) -> tuple[list, int | None]:
    """
    Run the output-weighted adaptive sampling loop.

    By default builds Mg-Ca-Zn Ensemble v3 via build_ensemble and
    applies utils.inverse_transform_outputs. Pass model_factory and
    inverse_transform to use a different system (e.g. In718).

    model_factory signature:
        fn(ckpt_path, scaling_path, n_folds) -> fitted-ensemble-object
    The returned object must implement:
        .fit(X, y)
        .predict(X)       -> np.ndarray (n, n_outputs) in log space
        .base_predict(X)  -> dict {name: np.ndarray}

    inverse_transform signature:
        fn(y_t: np.ndarray) -> np.ndarray  (physical space)

    Termination conditions (whichever comes first):
      1. All outputs simultaneously exceed ACCURACY_TARGET, AND
         total selected >= target_n (if supplied).
      2. max_rounds reached.
      3. Candidate pool exhausted.

    When target_n is supplied the batch is clipped to land exactly on
    target_n before convergence is checked.

    Parameters
    ----------
    X_pool           : (n_pool, d)   raw input pool
    y_t_pool         : (n_pool, p)   log-transformed output pool
    y_pool           : (n_pool, p)   physical-space output pool
    X_test           : (n_test, d)   held-out test inputs
    y_test           : (n_test, p)   held-out test outputs (physical)
    models_dir       : directory for PINN checkpoints / scaling files
    initial_n        : number of randomly selected seed points
    batch_size       : points added per round
    max_rounds       : safety cap on number of rounds
    target_n         : if set, enforce n >= target_n before convergence
    label            : string tag for checkpoint filenames
    model_factory    : callable that builds the ensemble (see above)
    train_fn         : callable(X, y, ckpt_path, scaling_path) that
                       trains the PINN (default: model.train_pinn)
    inverse_transform: callable that converts log predictions to
                       physical space (default: utils version)
    n_outputs        : number of outputs (inferred from y_test if None)

    Returns
    -------
    history   : list of dicts {n, acc, ensemble} one per round
    threshold : sample count at convergence, or None if not reached
    """
    # Defaults for Mg-Ca-Zn
    if train_fn is None:
        train_fn = train_pinn

    if model_factory is None:
        def model_factory(ckpt_path, scaling_path, n_folds):
            return build_ensemble(ckpt_path, scaling_path, n_folds)

    if inverse_transform is None:
        inverse_transform = inverse_transform_outputs

    if n_outputs is None:
        n_outputs = y_test.shape[1]

    rng        = np.random.default_rng(RAND_SEED)
    selected   = rng.choice(len(X_pool), initial_n, replace=False).tolist()
    candidates = [i for i in range(len(X_pool)) if i not in set(selected)]

    history   = []
    threshold = None

    for rnd in range(1, max_rounds + 1):
        ia     = np.array(selected)
        n      = len(selected)
        X_tr   = X_pool[ia]
        y_t_tr = y_t_pool[ia]

        print(f"\n  Round {rnd} | n={n}")

        ckpt_path    = models_dir / f'pinn_{label}_{n}.pt'
        scaling_path = models_dir / f'scaling_{label}_{n}.npz'

        # Train PINN using the provided or default training function
        train_fn(X_tr, y_t_tr, ckpt_path, scaling_path)

        ens = model_factory(
            ckpt_path, scaling_path,
            min(N_FOLDS, max(3, n // 150)),
        )
        ens.fit(X_tr, y_t_tr)

        y_pred = inverse_transform(ens.predict(X_test))
        acc    = compute_accuracy(y_test, y_pred)
        n_pass = int(np.sum(acc >= ACCURACY_TARGET))
        history.append({'n': n, 'acc': acc.copy(), 'ensemble': ens})

        print(f"  Mean={acc.mean():.2f}%  Pass={n_pass}/{n_outputs}")
        output_labels = (OUTPUT_NAMES if n_outputs == len(OUTPUT_NAMES)
                         else [f'output_{i}' for i in range(n_outputs)])
        for i, name in enumerate(output_labels):
            status = 'PASS' if acc[i] >= ACCURACY_TARGET else 'FAIL'
            print(f"    {name:<12} {acc[i]:.2f}%  {status}")

        at_target = (target_n is None) or (n >= target_n)
        if np.all(acc >= ACCURACY_TARGET) and at_target:
            threshold = n
            print(f"\n  Converged at n={n}")
            np.save(models_dir / f'indices_{label}_{n}.npy',
                    np.array(selected))
            break

        if not candidates:
            print("  Candidate pool exhausted.")
            break

        weights     = compute_output_weights(acc)
        ca          = np.array(candidates)
        bp          = np.stack(
            list(ens.base_predict(X_pool[ca]).values()), axis=0)
        uncertainty = (bp.var(axis=0) * weights).sum(axis=1)

        if target_n is not None and n < target_n:
            this_batch = min(batch_size, target_n - n)
        else:
            this_batch = batch_size

        new_idx    = ca[diverse_batch_select(
            uncertainty, X_pool[ca], this_batch)].tolist()
        selected  += new_idx
        added      = set(new_idx)
        candidates = [i for i in candidates if i not in added]

    return history, threshold