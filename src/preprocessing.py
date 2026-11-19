"""
preprocessing.py: Clean, transform, and prepare encrypted network traffic data.

Creates:
  1. app_label: 7 broad application categories (BROWSING, VOIP, FILE_TRANSFER, P2P, CHAT, MAIL, STREAMING)
  2. is_vpn: Binary flag (1 if encapsulated in VPN tunnel, 0 otherwise)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

FEATURE_COLUMNS = [
    'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat',
    'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond',
    'flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat',
    'std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active',
    'min_idle', 'mean_idle', 'max_idle', 'std_idle'
]

def map_app_label(raw_label: str) -> str:
    """Normalize raw class name to clean application category."""
    clean = str(raw_label).upper().replace('VPN-', '')
    if clean == 'FT':
        return 'FILE_TRANSFER'
    return clean

def load_and_preprocess_data(csv_path: str = "consolidated_traffic_data.csv"):
    print(f"[+] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Binary VPN Indicator
    df['is_vpn'] = df['traffic_type'].apply(lambda x: 1 if str(x).upper().startswith('VPN-') else 0)

    # 2. 7-Class Application Label
    df['app_label'] = df['traffic_type'].apply(map_app_label)

    print(f"[+] Loaded {len(df):,} flows.")
    print(f"    - VPN flows: {df['is_vpn'].sum():,} ({df['is_vpn'].mean()*100:.1f}%)")
    print(f"    - Non-VPN flows: {(df['is_vpn'] == 0).sum():,} ({(1 - df['is_vpn'].mean())*100:.1f}%)")
    print("    - 7-class distribution:\n", df['app_label'].value_counts().to_string())

    return df

def prepare_splits(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    """
    Stratified split into Train (70%), Validation (10%), and Test (20%).
    Keeps is_vpn intact and extracts scaled features.
    """
    X = df[FEATURE_COLUMNS].copy()
    y_app = df['app_label'].copy()
    y_vpn = df['is_vpn'].copy()

    # Encode application labels to integers
    label_encoder = LabelEncoder()
    y_app_encoded = label_encoder.fit_transform(y_app)

    # Stratify by combined (app_label + is_vpn) to guarantee balanced representation
    stratify_group = df['traffic_type']

    # Step 1: Split out test set (20%)
    X_train_val, X_test, y_app_tv, y_app_test, y_vpn_tv, y_vpn_test, strat_tv, strat_test = train_test_split(
        X, y_app_encoded, y_vpn, stratify_group,
        test_size=test_size, random_state=random_state, stratify=stratify_group
    )

    # Step 2: Split train_val into train (7/8 = 70% total) and validation (1/8 = 10% total)
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_app_train, y_app_val, y_vpn_train, y_vpn_val = train_test_split(
        X_train_val, y_app_tv, y_vpn_tv,
        test_size=val_relative_size, random_state=random_state, stratify=strat_tv
    )

    # Fit scaler strictly on training set to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save artifacts for inference and evaluation
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    # Save prepared dataset partitions
    np.savez_compressed(
        "data/prepared_data.npz",
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_app_train=y_app_train,
        y_app_val=y_app_val,
        y_app_test=y_app_test,
        y_vpn_train=y_vpn_train.values,
        y_vpn_val=y_vpn_val.values,
        y_vpn_test=y_vpn_test.values,
        classes=label_encoder.classes_,
        feature_names=np.array(FEATURE_COLUMNS)
    )

    print("\n[+] Data split complete:")
    print(f"    - Training set:   {X_train.shape[0]:,} samples (70%)")
    print(f"    - Validation set: {X_val.shape[0]:,} samples (10%)")
    print(f"    - Test set:       {X_test.shape[0]:,} samples (20%)")
    print(f"[+] Saved artifacts to 'models/' and 'data/prepared_data.npz'")

if __name__ == "__main__":
    df = load_and_preprocess_data()
    prepare_splits(df)
