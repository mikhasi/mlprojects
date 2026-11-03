
# 1

import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.datasets import fetch_openml

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_or_download_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "adult.csv")
    
    if not os.path.exists(csv_path):
        print(f"Dataset missing in {data_dir}. Downloading source dataset from OpenML...")
        adult = fetch_openml(name='adult', version=2, as_frame=True)
        df = adult.frame
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
    else:
        print(f"Loading dataset from local {csv_path}")
        df = pd.read_csv(csv_path)
        
    target_col = adult.target_names[0] if 'adult' in locals() and hasattr(adult, 'target_names') else df.columns[-1]
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    if y.dtype == object:
        y = np.array([1 if val.strip('.').endswith('>50K') else 0 for val in y])
        
    return X, y

if __name__ == "__main__":
    set_seeds(42)
    X, y = load_or_download_data()
    print(f"Data loaded successfully. Features shape: {X.shape}, Target shape: {y.shape}")