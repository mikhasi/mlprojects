
# 5

import os
import random
import numpy as np
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        if 'capital-gain' in X_df.columns and 'capital-loss' in X_df.columns:
            X_df['capital_net'] = X_df['capital-gain'] - X_df['capital-loss']
        if 'hours-per-week' in X_df.columns and 'age' in X_df.columns:
            denominator = (X_df['age'] - 16).replace(0, 1)
            X_df['hours_per_age'] = X_df['hours-per-week'] / denominator
        return X_df

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

def run_pipeline():
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
    
    train_df = X_train.copy()
    train_df['target'] = y_train
    initial_len = len(train_df)
    train_df = train_df.drop_duplicates()
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target'].values
    print(f"[Data Preparation] Dropped {initial_len - len(train_df)} duplicate rows from training set.")

    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    X_train[categorical_features] = X_train[categorical_features].replace('?', np.nan)
    X_test[categorical_features] = X_test[categorical_features].replace('?', np.nan)

    base_numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    engineered_numeric_features = base_numeric_features + ['capital_net', 'hours_per_age']

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    print("\n--- Running Baseline: Without Engineered Features ---")
    num_transformer_base = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', PercentileClipper()),
        ('scaler', StandardScaler())
    ])
    preprocessor_base = ColumnTransformer([
        ('num', num_transformer_base, base_numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])
    
    model_base = Pipeline([
        ('preprocessor', preprocessor_base),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    model_base.fit(X_train, y_train)
    y_pred_base = model_base.predict(X_test)
    
    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base, pos_label=1)

    print("--- Running Baseline: With Engineered Features ---")
    num_transformer_eng = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', PercentileClipper()),
        ('scaler', StandardScaler())
    ])
    preprocessor_eng = Pipeline([
        ('engineer', DerivedFeatureTransformer()),
        ('col_transform', ColumnTransformer([
            ('num', num_transformer_eng, engineered_numeric_features),
            ('cat', cat_transformer, categorical_features)
        ]))
    ])
    
    model_eng = Pipeline([
        ('preprocessor', preprocessor_eng),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    model_eng.fit(X_train, y_train)
    y_pred_eng = model_eng.predict(X_test)
    
    acc_eng = accuracy_score(y_test, y_pred_eng)
    f1_eng = f1_score(y_test, y_pred_eng, pos_label=1)

    print("\n==========================================================")
    print(" TASK 5: FEATURE ENGINEERING & LOGISTIC REGRESSION SUMMARY ")
    print("==========================================================")
    print(f"{'Pipeline Configuration':<32} | {'Accuracy':<10} | {'F1-Score (>50K)':<15}")
    print("-" * 63)
    print(f"{'Without Engineered Features':<32} | {acc_base:.4f}     | {f1_base:.4f}")
    print(f"{'With Engineered Features':<32} | {acc_eng:.4f}     | {f1_eng:.4f}")
    print("==========================================================")

if __name__ == "__main__":
    run_pipeline()