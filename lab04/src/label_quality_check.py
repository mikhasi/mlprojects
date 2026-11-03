
# 6

import os
import random
import numpy as np
import torch
import pandas as pd

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def audit_label_quality():
    set_seeds(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    csv_path = os.path.join(root_dir, "data", "adult.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset missing at {csv_path}. Run ingestion first.")
        
    df = pd.read_csv(csv_path)
    
    if 'class' in df.columns and 'income' not in df.columns:
        df = df.rename(columns={'class': 'income'})

    print("==========================================")
    print("1. DUPLICATE-CONFLICT AUDIT")
    print("==========================================")
    
    feature_cols = [col for col in df.columns if col != 'income']
    
    grouped = df.groupby(feature_cols)['income'].nunique()
    conflicting_groups = grouped[grouped > 1]
    
    if len(conflicting_groups) > 0:
        conflicting_keys = conflicting_groups.index
        conflict_rows = df.set_index(feature_cols).loc[conflicting_keys]
        conflict_row_count = len(conflict_rows)
        
        print(f"  [RESULT] Found {len(conflicting_groups)} unique feature combinations with conflicting income labels.")
        print(f"  [RESULT] Total rows affected by label conflicts: {conflict_row_count}")
    else:
        print("  [PASS] Zero duplicate feature rows with conflicting income labels found.")

    print("\n==========================================")
    print("2. MANUAL DATA DICTIONARY RE-CHECK (50 Random Rows)")
    print("==========================================")
    
    sample_df = df.sample(n=50, random_state=42)
    
    violations_found = 0
    for idx, row in sample_df.iterrows():
        issues = []
        
        if not (17 <= row['age'] <= 90):
            issues.append(f"age ({row['age']})")
        if not (1 <= row['hours-per-week'] <= 99):
            issues.append(f"hours-per-week ({row['hours-per-week']})")
        if not (0 <= row['capital-gain'] <= 99999):
            issues.append(f"capital-gain ({row['capital-gain']})")
        if not (0 <= row['capital-loss'] <= 4356):
            issues.append(f"capital-loss ({row['capital-loss']})")
            
        if issues:
            violations_found += 1
            print(f"  [VIOLATION] Row Index {idx}: Out-of-range values in {issues}")

    if violations_found == 0:
        print("  [PASS] All 50 randomly sampled rows conform strictly to data dictionary constraints.")
    else:
        print(f"  [WARNING] Found {violations_found} rows with range violations in the 50-row sample.")

    print("==========================================")

if __name__ == "__main__":
    audit_label_quality()