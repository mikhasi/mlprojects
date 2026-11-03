
# 3

import os
import pandas as pd

def validate_schema():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    csv_path = os.path.join(root_dir, "data", "adult.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset missing at {csv_path}. Run ingestion first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path}. Shape: {df.shape}\n")
    
    if 'class' in df.columns and 'income' not in df.columns:
        df = df.rename(columns={'class': 'income'})

    int_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
    
    ranges = {
        'age': (17, 90),
        'hours-per-week': (1, 99),
        'capital-gain': (0, 99999),
        'capital-loss': (0, 4356)
    }
    
    violations = []
    
    print("==========================================")
    print("1. CHECKING COLUMN TYPES & INTEGER PARSING")
    print("==========================================")
    for col in int_cols:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            invalid_mask = converted.isna() & df[col].notna()
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                violations.append(f"Type Violation: Column '{col}' contains {invalid_count} non-integer values.")
            else:
                print(f"  [PASS] '{col}' conforms to integer expectations.")

    print("\n==========================================")
    print("2. CHECKING CATEGORICAL NULL POLICY ('?')")
    print("==========================================")
    for col in cat_cols:
        if col in df.columns:
            q_marks = (df[col].astype(str).str.strip() == '?').sum()
            print(f"  [INFO] '{col}': {q_marks} missing values ('?') detected (Allowed by policy).")

    print("\n==========================================")
    print("3. CHECKING VALUE RANGES")
    print("==========================================")
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            out_of_range = df[(numeric_vals < min_val) | (numeric_vals > max_val)]
            count_oor = len(out_of_range)
            if count_oor > 0:
                violations.append(f"Range Violation: Column '{col}' has {count_oor} rows outside range [{min_val}, {max_val}].")
                print(f"  [FAIL] '{col}': {count_oor} values outside [{min_val}, {max_val}].")
            else:
                print(f"  [PASS] '{col}': All values within range [{min_val}, {max_val}].")

    print("\n==========================================")
    print("4. VALIDATION SUMMARY")
    print("==========================================")
    if violations:
        print(f"Validation completed with {len(violations)} violation category/categories:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("All validation checks passed successfully with zero violations!")
        
    print(f"Total rows preserved: {len(df)} (No rows were silently dropped).")
    return df, violations

if __name__ == "__main__":
    validate_schema()