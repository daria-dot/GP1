import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # up to project root

def load_data() -> pd.DataFrame:
    """
    Load raw input and output data from data/raw/ and return as a combined DataFrame.
    
    Returns
    -------
    pd.DataFrame
        75000 rows × 10 columns: T, X_Ca, X_Zn, X_Mg, mu_Ca, mu_Mg, mu_Zn,
        Dv_Ca, Dv_Mg, Dv_Zn, Vm
    """
    input_data = pd.read_csv(
        BASE_DIR / 'data' / 'raw' / 'input_data.txt',
        sep=r'\s*,\s*', engine='python',
        header=None, names=['T', 'X_Ca', 'X_Zn']
    )
    output_data = pd.read_csv(
        BASE_DIR / 'data' / 'raw' / 'output_data.txt',
        sep=r'\s+', engine='python',
        header=None, names=['mu_Ca', 'mu_Mg', 'mu_Zn', 'Dv_Ca', 'Dv_Mg', 'Dv_Zn', 'Vm']
    )

    data = pd.concat([input_data, output_data], axis=1)

    # Derive implicit Mg mole fraction
    data['X_Mg'] = 1 - data['X_Ca'] - data['X_Zn']

    # Reorder columns logically
    data = data[['T', 'X_Ca', 'X_Zn', 'X_Mg',
                 'mu_Ca', 'mu_Mg', 'mu_Zn',
                 'Dv_Ca', 'Dv_Mg', 'Dv_Zn', 'Vm']]

    print(f"[data_loader] Loaded {len(data)} rows, {data.shape[1]} columns")
    return data