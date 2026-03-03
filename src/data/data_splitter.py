"""
data_splitter.py

Responsibilities:
  - Load transformed_dataset.csv
  - Split into train / validation / test sets (70 / 15 / 15 by default)
  - Refit StandardScaler on TRAINING SET ONLY (no leakage)
  - Update scaling_parameters.json with train-only scaler params
  - Save train_set.csv, validation_set.csv, test_set.csv

Run this script directly:
    python src/data/data_splitter.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
METADATA_DIR  = BASE_DIR / 'data' / 'metadata'

INPUT_COLS            = ['T', 'X_Ca', 'X_Zn']
TRANSFORMED_OUT_COLS  = ['mu_Ca', 'mu_Mg', 'mu_Zn',
                          'log_Dv_Ca', 'log_Dv_Mg', 'log_Dv_Zn', 'Vm']


def split_data(
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    random_seed: int   = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the transformed dataset into train / validation / test sets.

    Parameters
    ----------
    train_ratio  : fraction of data for training   (default 0.70)
    val_ratio    : fraction of data for validation  (default 0.15)
    test_ratio   : fraction of data for test        (default 0.15)
    random_seed  : random seed for reproducibility  (default 42)

    Returns
    -------
    train, val, test : pd.DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "train + val + test ratios must sum to 1.0"

    # Load transformed data
    transformed_path = PROCESSED_DIR / 'transformed_dataset.csv'
    if not transformed_path.exists():
        raise FileNotFoundError(
            "transformed_dataset.csv not found. "
            "Run data_preprocessor.py first."
        )
    data = pd.read_csv(transformed_path)
    print(f"[splitter] Loaded transformed_dataset.csv  ({len(data)} rows)")

    # ------------------------------------------------------------------
    # Split: first carve out test set, then split remainder into train/val
    # ------------------------------------------------------------------
    test_size      = test_ratio
    val_size_adj   = val_ratio / (train_ratio + val_ratio)  # relative to remainder

    train_val, test = train_test_split(
        data, test_size=test_size, random_state=random_seed, shuffle=True
    )
    train, val = train_test_split(
        train_val, test_size=val_size_adj, random_state=random_seed, shuffle=True
    )

    print(f"[splitter] Split sizes:")
    print(f"  Train:      {len(train):>6} rows  ({len(train)/len(data)*100:.1f}%)")
    print(f"  Validation: {len(val):>6} rows  ({len(val)/len(data)*100:.1f}%)")
    print(f"  Test:       {len(test):>6} rows  ({len(test)/len(data)*100:.1f}%)")

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    train.to_csv(PROCESSED_DIR / 'train_set.csv',      index=False)
    val.to_csv(  PROCESSED_DIR / 'validation_set.csv', index=False)
    test.to_csv( PROCESSED_DIR / 'test_set.csv',       index=False)
    print(f"[splitter] Saved train_set.csv, validation_set.csv, test_set.csv")

    # ------------------------------------------------------------------
    # Refit scalers on TRAINING DATA ONLY and update scaling_parameters.json
    # This is the scaler that must be used for all model training/inference
    # ------------------------------------------------------------------
    input_scaler  = StandardScaler().fit(train[INPUT_COLS])
    output_scaler = StandardScaler().fit(train[TRANSFORMED_OUT_COLS])

    scaling_path = METADATA_DIR / 'scaling_parameters.json'
    with open(scaling_path, 'r') as f:
        scaling_params = json.load(f)

    scaling_params['warning'] = (
        'Scalers below are fit on the TRAINING SET ONLY (no data leakage). '
        'Always use these params for model training and inference.'
    )
    scaling_params['train_split_info'] = {
        'train_size':      len(train),
        'val_size':        len(val),
        'test_size':       len(test),
        'total':           len(data),
        'random_seed':     random_seed,
        'train_ratio':     train_ratio,
        'val_ratio':       val_ratio,
        'test_ratio':      test_ratio
    }
    scaling_params['inputs_train_only'] = {
        col: {'mean': float(input_scaler.mean_[i]),
              'std':  float(input_scaler.scale_[i])}
        for i, col in enumerate(INPUT_COLS)
    }
    scaling_params['outputs_train_only'] = {
        col: {'mean': float(output_scaler.mean_[i]),
              'std':  float(output_scaler.scale_[i])}
        for i, col in enumerate(TRANSFORMED_OUT_COLS)
    }

    with open(scaling_path, 'w') as f:
        json.dump(scaling_params, f, indent=2)
    print(f"[splitter] Updated scaling_parameters.json with train-only scaler params")

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    print(f"\n[splitter] Sanity checks:")
    print(f"  No index overlap train/val:  "
          f"{len(set(train.index) & set(val.index)) == 0}")
    print(f"  No index overlap train/test: "
          f"{len(set(train.index) & set(test.index)) == 0}")
    print(f"  No index overlap val/test:   "
          f"{len(set(val.index) & set(test.index)) == 0}")

    # Check input range coverage is similar across splits
    for col in INPUT_COLS:
        t_range = (train[col].min(), train[col].max())
        v_range = (val[col].min(),   val[col].max())
        s_range = (test[col].min(),  test[col].max())
        print(f"  {col} range — train: [{t_range[0]:.4e}, {t_range[1]:.4e}]  "
              f"val: [{v_range[0]:.4e}, {v_range[1]:.4e}]  "
              f"test: [{s_range[0]:.4e}, {s_range[1]:.4e}]")

    print(f"\n[splitter] Splitting complete.")
    return train, val, test


if __name__ == '__main__':
    split_data()