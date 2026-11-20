"""
Theoria — train_vpn.py
Encrypted Network Traffic Analysis & Classification Engine.

Trains and evaluates the binary VPN Encapsulation Classifier:
Distinguishes between Direct Internet Traffic and VPN-Encapsulated Traffic
using side-channel timing and rate metadata.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

def train_vpn_model(data_path="data/prepared_data.npz", output_dir="models", results_dir="results"):
    print("[+] Loading prepared dataset partitions for VPN classification...")
    data = np.load(data_path, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_vpn_train']
    X_test, y_test = data['X_test'], data['y_vpn_test']

    print(f"[+] Loaded: Train = {len(X_train):,}, Test = {len(X_test):,}")
    print(f"    - Train class balance: Direct={np.sum(y_train==0):,} ({(y_train==0).mean()*100:.1f}%), VPN={np.sum(y_train==1):,} ({(y_train==1).mean()*100:.1f}%)")

    print("\n[>] Training Random Forest VPN Encapsulation Classifier (100 trees)...")
    rf_vpn = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_vpn.fit(X_train, y_train)

    print("[>] Evaluating on 11,942 unseen test flows...")
    y_pred = rf_vpn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"[✓] VPN Classification Accuracy: {acc * 100:.2f}% | F1-Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    target_names = ["DIRECT (Non-VPN)", "VPN TUNNEL"]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "vpn_classifier.pkl")
    joblib.dump(rf_vpn, model_path)
    print(f"[✓] Saved model artifact to {model_path}")

    # Generate publication-grade confusion matrix
    os.makedirs(results_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 11, "weight": "bold"}, ax=ax,
                linewidths=0.5, linecolor='#F1F5F9')

    ax.set_title(f"Theoria: VPN Encapsulation Detection\nAccuracy: {acc*100:.2f}% | F1: {f1:.4f} (N = {len(y_test):,} Flows)",
                 fontsize=12, fontweight='bold', pad=15, color='#0F172A')
    ax.set_ylabel('True Transport Mode', fontsize=10.5, fontweight='bold', color='#334155')
    ax.set_xlabel('Predicted Transport Mode', fontsize=10.5, fontweight='bold', color='#334155')

    plt.tight_layout()
    cm_path = os.path.join(results_dir, "vpn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[✓] Saved confusion matrix to {cm_path}")

    return rf_vpn, acc, f1

if __name__ == "__main__":
    train_vpn_model()
