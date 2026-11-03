
# 4

import os
import random
import numpy as np
import torch
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            self.lower_bounds[col] = X_df[col].quantile(0.01)
            self.upper_bounds[col] = X_df[col].quantile(0.99)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.lower_bounds:
                X_df[col] = X_df[col].clip(lower=self.lower_bounds[col], upper=self.upper_bounds[col])
        return X_df.values

def clean_and_transform_data():
    set_seeds(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    csv_path = os.path.join(root_dir, "data", "adult.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset missing at {csv_path}. Run ingestion first.")
        
    df = pd.read_csv(csv_path)
    
    if 'class' in df.columns and 'income' not in df.columns:
        df = df.rename(columns={'class': 'income'})

    if df['income'].dtype == object:
        df['income'] = df['income'].apply(lambda val: 1 if str(val).strip('.').endswith('>50K') else 0)

    X = df.drop(columns=['income'])
    y = df['income'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Initial Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    train_df = X_train.copy()
    train_df['target'] = y_train
    initial_train_len = len(train_df)
    train_df = train_df.drop_duplicates()
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target'].values
    print(f"Dropped {initial_train_len - len(train_df)} exact duplicate rows from training set.")
    print(f"Cleaned Train shape: {X_train.shape}")

    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    X_train[categorical_features] = X_train[categorical_features].replace('?', np.nan)
    X_test[categorical_features] = X_test[categorical_features].replace('?', np.nan)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', PercentileClipper())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Transformed Train shape: {X_train_transformed.shape}")
    print(f"Transformed Test shape: {X_test_transformed.shape}")

    results_dir = os.path.join(script_dir, "../results")
    os.makedirs(results_dir, exist_ok=True)
    preprocessor_path = os.path.join(results_dir, "preprocessor.joblib")
    
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Fitted preprocessor successfully saved to {preprocessor_path}")

    return X_train_transformed, X_test_transformed, y_train, y_test

if __name__ == "__main__":
    clean_and_transform_data()