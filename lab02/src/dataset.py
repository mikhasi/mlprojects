import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def load_or_download_digits():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "digits.csv")
    
    if not os.path.exists(csv_path):
        print(f"Dataset missing in {data_dir}. Downloading source dataset...")
        X, y = load_digits(return_X_y=True)
        df = pd.DataFrame(X)
        df["target"] = y
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
    else:
        print(f"Loading dataset from local {csv_path}")
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        
    X = X / 16.0
    
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_or_download_digits()
    print(f"Training data shape (X, y): {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape (X, y): {X_test.shape}, {y_test.shape}")