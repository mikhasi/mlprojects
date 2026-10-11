import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

def main():
    set_seeds(42)

    X, y = load_or_download_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)

    os.makedirs("results", exist_ok=True)
    os.makedirs("lab01/results", exist_ok=True)
    
    output_path = "results/baseline_accuracy.csv"

    results_data = {
        "model": ["LogisticRegression", "RandomForestClassifier"],
        "accuracy": [round(lr_accuracy, 4), round(rf_accuracy, 4)]
    }

    df = pd.DataFrame(results_data)
    df.to_csv(output_path, index=False)

    print("Baseline models trained successfully!")
    print(df)

if __name__ == "__main__":
    main()