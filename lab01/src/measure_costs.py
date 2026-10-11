import os
import random
import time
import joblib
import pandas as pd
import numpy as np
import torch
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    csv_path = os.path.join(data_dir, "breast_cancer.csv")
    
    if not os.path.exists(csv_path):
        print(f"Dataset missing in {data_dir}. Downloading source dataset...")
        raw_data = load_breast_cancer(as_frame=True)
        df = raw_data.frame
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
    else:
        print(f"Loading dataset from local {csv_path}")
        df = pd.read_csv(csv_path)
        
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return X, y

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    set_seeds(42)

    X, y = load_or_download_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    single_sample = X_test[0:1]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    os.makedirs("results", exist_ok=True)

    for name, model in models.items():
        print(f"Profiling system costs for {name}...")

        model.fit(X_train, y_train)

        train_times = []
        train_mem_peaks = []

        for _ in range(5):
            mem_before = get_process_memory_mb()
            start_time = time.perf_counter()
            
            model.fit(X_train, y_train)
            
            end_time = time.perf_counter()
            mem_after = get_process_memory_mb()
            
            train_times.append(end_time - start_time)
            train_mem_peaks.append(max(mem_before, mem_after))

        median_train_time = np.median(train_times)
        peak_train_memory = np.max(train_mem_peaks)

        model.predict(single_sample)

        inf_times = []
        inf_mem_peaks = []

        for _ in range(100):
            mem_before = get_process_memory_mb()
            start_time = time.perf_counter()
            
            model.predict(single_sample)
            
            end_time = time.perf_counter()
            mem_after = get_process_memory_mb()
            
            inf_times.append(end_time - start_time)
            inf_mem_peaks.append(max(mem_before, mem_after))

        median_inf_time = np.median(inf_times)
        peak_inf_memory = np.max(inf_mem_peaks)

        model_filename = f"results/{name.lower()}_model.joblib"
        joblib.dump(model, model_filename)
        
        size_bytes = os.path.getsize(model_filename)
        size_kb = size_bytes / 1024.0

        results.append({
            "model": name,
            "median_train_time_sec": round(median_train_time, 6),
            "peak_train_memory_mib": round(peak_train_memory, 2),
            "median_inference_time_sec": round(median_inf_time, 8),
            "peak_inference_memory_mib": round(peak_inf_memory, 2),
            "model_size_bytes": size_bytes,
            "model_size_kb": round(size_kb, 2)
        })

    df = pd.DataFrame(results)
    df.to_csv("results/system_costs.csv", index=False)
    print("\nSystem cost profiling summary:")
    print(df.to_string())

if __name__ == "__main__":
    main()