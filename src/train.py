"""
train.py: Multi-model training and benchmarking for Encrypted Network Traffic Analysis.

Model Evaluation Order:
  1. Dummy Classifier (sanity floor ~ 33%)
  2. Decision Tree (paper baseline - gate check: expected ~80-90%)
  3. k-Nearest Neighbors (paper baseline - gate check: expected ~80-90%)
  4. Random Forest (ensemble baseline)
  5. Gradient Boosted Trees (XGBoost / HistGradientBoosting)
  6. Multi-Layer Perceptron (neural network)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

# Attempt XGBoost import (falls back to HistGradientBoosting if OpenMP / libomp is missing on macOS)
XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def load_data(npz_path="data/prepared_data.npz"):
    data = np.load(npz_path, allow_pickle=True)
    return (
        data['X_train'], data['y_app_train'],
        data['X_val'], data['y_app_val'],
        data['X_test'], data['y_app_test'],
        data['classes'], data['feature_names']
    )

def plot_confusion_matrix(y_true, y_pred, classes, model_name, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Normalized Confusion Matrix - {model_name}")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"{model_name.lower().replace(' ', '_').replace('/', '_')}_confusion_matrix.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def evaluate_model(model, X_train, y_train, X_test, y_test, classes, model_name):
    print(f"\n=======================================================")
    print(f"[>] Training: {model_name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"[✓] {model_name} Performance on Test Set:")
    print(f"    - Accuracy:        {acc * 100:.2f}%")
    print(f"    - Macro F1:        {macro_f1:.4f}")
    print(f"    - Weighted F1:     {weighted_f1:.4f}")
    
    print("\nDetailed Per-Class Report:")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))
    
    # Save confusion matrix plot
    plot_confusion_matrix(y_test, y_pred, classes, model_name)
    
    # Save model artifact
    clean_name = model_name.lower().replace(' ', '_').replace('/', '_')
    model_save_path = f"models/{clean_name}.pkl"
    joblib.dump(model, model_save_path)
    
    return {
        "model": model_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }

def main():
    print("[+] Loading prepared dataset partitions...")
    X_train, y_train, X_val, y_val, X_test, y_test, classes, feature_names = load_data()
    print(f"[+] Loaded: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    print(f"[+] Application Classes ({len(classes)}): {list(classes)}")
    
    results = []
    
    # -------------------------------------------------------------
    # 1. Dummy Classifier (Sanity Floor: Predicts most frequent class)
    # -------------------------------------------------------------
    dummy = DummyClassifier(strategy="most_frequent")
    res_dummy = evaluate_model(dummy, X_train, y_train, X_test, y_test, classes, "Dummy Classifier")
    results.append(res_dummy)
    
    # -------------------------------------------------------------
    # 2. Decision Tree (Paper Model - Gate Check 1)
    # -------------------------------------------------------------
    dt = DecisionTreeClassifier(random_state=42)
    res_dt = evaluate_model(dt, X_train, y_train, X_test, y_test, classes, "Decision Tree")
    results.append(res_dt)
    
    # -------------------------------------------------------------
    # 3. k-Nearest Neighbors (Paper Model - Gate Check 2)
    # -------------------------------------------------------------
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    res_knn = evaluate_model(knn, X_train, y_train, X_test, y_test, classes, "k-Nearest Neighbors")
    results.append(res_knn)
    
    # Gate Check Assessment
    print("\n" + "#"*60)
    print(" [GATE CHECK VERIFICATION]")
    print(f"  Decision Tree Accuracy:       {res_dt['accuracy']*100:.2f}% | Macro F1: {res_dt['macro_f1']:.4f}")
    print(f"  k-NN (k=5) Accuracy:          {res_knn['accuracy']*100:.2f}% | Macro F1: {res_knn['macro_f1']:.4f}")
    print("#"*60)
    
    if res_dt['accuracy'] < 0.70 or res_knn['accuracy'] < 0.70:
        print("[!] WARNING: Gate check failed! Accuracy is significantly below expected paper reproduction (~80-90%). Halting before training ensembles.")
        return
    else:
        print("[✓] Gate check passed! Both paper models reproduced strong performance. Proceeding to ensembles.")
    
    # -------------------------------------------------------------
    # 4. Random Forest (Real Baseline)
    # -------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    res_rf = evaluate_model(rf, X_train, y_train, X_test, y_test, classes, "Random Forest")
    results.append(res_rf)
    
    # -------------------------------------------------------------
    # 5. Gradient Boosted Trees (XGBoost or HistGradientBoosting)
    # -------------------------------------------------------------
    if XGB_AVAILABLE:
        boost = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        res_boost = evaluate_model(boost, X_train, y_train, X_test, y_test, classes, "XGBoost")
    else:
        print("\n[*] Note: XGBoost C-library requires libomp on macOS. Using scikit-learn's native HistGradientBoostingClassifier (LightGBM-inspired histogram gradient boosting).")
        boost = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=6, random_state=42)
        res_boost = evaluate_model(boost, X_train, y_train, X_test, y_test, classes, "Gradient Boosting (HistGB)")
    results.append(res_boost)
    
    # -------------------------------------------------------------
    # 6. Multi-Layer Perceptron (Neural Net - Optional)
    # -------------------------------------------------------------
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42, early_stopping=True)
    res_mlp = evaluate_model(mlp, X_train, y_train, X_test, y_test, classes, "Multi-Layer Perceptron")
    results.append(res_mlp)
    
    # Save overall benchmarking summary
    with open("results/model_comparison.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*60)
    print("FINAL BENCHMARK COMPARISON SUMMARY:")
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<30} | {r['accuracy']*100:>8.2f}% | {r['macro_f1']:>10.4f} | {r['weighted_f1']:>10.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
