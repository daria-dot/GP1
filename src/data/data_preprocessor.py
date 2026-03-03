"""
data_preprocessor.py

Responsibilities:
  1. Save combined_dataset.csv       (raw merged data)
  2. Save transformed_dataset.csv    (log10 D outputs, inputs as-is)
  3. Save input_bounds.json          (x_L and x_U per input)
  4. Save output_statistics.json     (mean, std, min, max per output)
  5. Save scaling_parameters.json    (StandardScaler params for inputs + log-outputs)

Run this script directly:
    python src/data/data_preprocessor.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Project root is 3 levels up from this file (src/data/data_preprocessor.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
METADATA_DIR  = BASE_DIR / 'data' / 'metadata'

INPUT_COLS  = ['T', 'X_Ca', 'X_Zn']
OUTPUT_COLS = ['mu_Ca', 'mu_Mg', 'mu_Zn', 'Dv_Ca', 'Dv_Mg', 'Dv_Zn', 'Vm']
D_COLS      = ['Dv_Ca', 'Dv_Mg', 'Dv_Zn']   # columns that get log10 transform
LOG_COLS    = ['log_Dv_Ca', 'log_Dv_Mg', 'log_Dv_Zn']


def preprocess(data: pd.DataFrame) -> None:
    """
    Run full preprocessing pipeline and save all outputs to disk.

    Parameters
    ----------
    data : pd.DataFrame
        Combined raw dataset from data_loader.load_data()
    """

    # ------------------------------------------------------------------
    # 1. Save combined_dataset.csv (raw, no transforms)
    # ------------------------------------------------------------------
    combined_path = PROCESSED_DIR / 'combined_dataset.csv'
    data.to_csv(combined_path, index=False)
    print(f"[preprocessor] Saved combined_dataset.csv  ({len(data)} rows)")

    # ------------------------------------------------------------------
    # 2. Build transformed_dataset.csv
    #    - log10 transform diffusion coefficients
    #    - inputs unchanged (scaling params saved separately)
    # ------------------------------------------------------------------
    transformed = data.copy()

    for d_col, log_col in zip(D_COLS, LOG_COLS):
        if (transformed[d_col] <= 0).any():
            raise ValueError(
                f"Column '{d_col}' contains non-positive values — "
                "log10 transform is undefined. Check your raw data."
            )
        transformed[log_col] = np.log10(transformed[d_col])

    # Drop raw D columns, keep log versions
    transformed = transformed.drop(columns=D_COLS)

    # Final column order
    transformed = transformed[
        ['T', 'X_Ca', 'X_Zn', 'X_Mg',
         'mu_Ca', 'mu_Mg', 'mu_Zn',
         'log_Dv_Ca', 'log_Dv_Mg', 'log_Dv_Zn',
         'Vm']
    ]

    transformed_path = PROCESSED_DIR / 'transformed_dataset.csv'
    transformed.to_csv(transformed_path, index=False)
    print(f"[preprocessor] Saved transformed_dataset.csv")

    # ------------------------------------------------------------------
    # 3. Save input_bounds.json
    # ------------------------------------------------------------------
    input_bounds = {}
    for col in INPUT_COLS:
        input_bounds[col] = {
            'x_L': float(data[col].min()),
            'x_U': float(data[col].max()),
            'units': 'K' if col == 'T' else 'mole fraction'
        }
    # Also record the implicit simplex constraint
    input_bounds['constraint'] = 'X_Ca + X_Zn <= 1  (X_Mg = 1 - X_Ca - X_Zn >= 0)'

    bounds_path = METADATA_DIR / 'input_bounds.json'
    with open(bounds_path, 'w') as f:
        json.dump(input_bounds, f, indent=2)
    print(f"[preprocessor] Saved input_bounds.json")

    # ------------------------------------------------------------------
    # 4. Save output_statistics.json
    #    Statistics on RAW outputs (before log transform) for reference,
    #    and on transformed outputs for model targets
    # ------------------------------------------------------------------
    output_stats = {}

    print("\n[preprocessor] Output statistics (raw):")
    for col in OUTPUT_COLS:
        output_stats[col] = {
            'mean':  float(data[col].mean()),
            'std':   float(data[col].std()),
            'min':   float(data[col].min()),
            'max':   float(data[col].max()),
            'units': _get_units(col),
            'note':  'raw scale'
        }
        print(f"  {col:>10}: mean={data[col].mean():.4e}, "
              f"std={data[col].std():.4e}, "
              f"range=[{data[col].min():.4e}, {data[col].max():.4e}]")

    print("\n[preprocessor] Output statistics (log10 D):")
    for d_col, log_col in zip(D_COLS, LOG_COLS):
        log_vals = transformed[log_col]
        output_stats[log_col] = {
            'mean':  float(log_vals.mean()),
            'std':   float(log_vals.std()),
            'min':   float(log_vals.min()),
            'max':   float(log_vals.max()),
            'units': 'log10(m²/s)',
            'note':  'log10-transformed — use this as model target'
        }
        print(f"  {log_col:>14}: mean={log_vals.mean():.4f}, "
              f"std={log_vals.std():.4f}, "
              f"range=[{log_vals.min():.4f}, {log_vals.max():.4f}]")

    stats_path = METADATA_DIR / 'output_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(output_stats, f, indent=2)
    print(f"\n[preprocessor] Saved output_statistics.json")

    # ------------------------------------------------------------------
    # 5. Save scaling_parameters.json
    #    Fit StandardScaler on inputs and on transformed outputs
    #    These must be fit on TRAINING data only — here we fit on full
    #    dataset for metadata purposes; data_splitter.py will refit on
    #    train split only before model training.
    # ------------------------------------------------------------------
    transformed_output_cols = ['mu_Ca', 'mu_Mg', 'mu_Zn',
                                'log_Dv_Ca', 'log_Dv_Mg', 'log_Dv_Zn', 'Vm']

    input_scaler  = StandardScaler().fit(data[INPUT_COLS])
    output_scaler = StandardScaler().fit(transformed[transformed_output_cols])

    scaling_params = {
        'warning': (
            'These scalers are fit on the FULL dataset for reference only. '
            'For model training, always refit scalers on the training split '
            'to avoid data leakage. See data_splitter.py.'
        ),
        'inputs': {
            col: {'mean': float(input_scaler.mean_[i]),
                  'std':  float(input_scaler.scale_[i])}
            for i, col in enumerate(INPUT_COLS)
        },
        'outputs': {
            col: {'mean': float(output_scaler.mean_[i]),
                  'std':  float(output_scaler.scale_[i])}
            for i, col in enumerate(transformed_output_cols)
        }
    }

    scaling_path = METADATA_DIR / 'scaling_parameters.json'
    with open(scaling_path, 'w') as f:
        json.dump(scaling_params, f, indent=2)
    print(f"[preprocessor] Saved scaling_parameters.json")
    print(f"\n[preprocessor] All preprocessing complete.")


def _get_units(col: str) -> str:
    if col.startswith('mu'):
        return 'J/mol'
    elif col.startswith('Dv'):
        return 'm²/s'
    elif col == 'Vm':
        return 'm³/mol'
    return ''


# ------------------------------------------------------------------
# Run directly
# ------------------------------------------------------------------
if __name__ == '__main__':
    from data_loader import load_data
    data = load_data()
    preprocess(data)