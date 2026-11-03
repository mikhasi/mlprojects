
# 7

import os
import sys
import hashlib
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ingestion import ingest_data
from validation import validate_schema
from engineer_and_evaluate import run_pipeline

def compute_file_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    
    print("\n==================================================")
    print("STEP 1: INGESTION (Reading raw data)")
    print("==================================================")
    df_raw = ingest_data()

    print("\n==================================================")
    print("STEP 2: VALIDATION (Schema & Range checks)")
    print("==================================================")
    _, violations = validate_schema()

    print("\n==================================================")
    print("STEP 3, 4 & 5: CLEANING, FEATURES & BASELINE")
    print("==================================================")
    run_pipeline()

    print("\n==================================================")
    print("STEP 7: COMPUTING HASH & UPDATING CONFIG")
    print("==================================================")
    
    target_file_path = os.path.join(root_dir, "data", "adult.csv")
    if not os.path.exists(target_file_path):
        target_file_path = os.path.join(root_dir, "data", "adult_processed.csv")
        
    file_hash = compute_file_sha256(target_file_path)
    print(f"Dataset SHA256 Hash: {file_hash}")

    config_path = os.path.join("../", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config['processed_dataset_sha256'] = file_hash
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        print(f"Updated {config_path} with new SHA256 hash.")

    print("\n==================================================")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()