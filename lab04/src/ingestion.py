
# 2

import os
import pandas as pd
from sklearn.datasets import fetch_openml

def ingest_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "adult.csv")
    
    if not os.path.exists(csv_path):
        print(f"Dataset missing in {data_dir}. Downloading from OpenML...")
        adult = fetch_openml(name='adult', version=2, as_frame=True)
        df = adult.frame
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
    else:
        print(f"Loading raw dataset from local {csv_path}...")
        df = pd.read_csv(csv_path)
        
    num_rows, num_cols = df.shape
    print(f"[Ingestion Log] Row count: {num_rows}")
    print(f"[Ingestion Log] Column count: {num_cols}")
    
    assert num_rows == 48842, f"Expected 48842 rows, but got {num_rows}"
    assert num_cols == 15, f"Expected 15 columns, but got {num_cols}"
    
    return df

if __name__ == "__main__":
    raw_df = ingest_data()
    print("Ingestion step completed successfully. Raw data unchanged.")