"""
Theoria — verify_pipeline.py
Encrypted Network Traffic Analysis & Classification Engine.

End-to-end system health and verification script.
Tests:
  1. File existence (raw data, prepared partitions, model weights, evaluation plots).
  2. Model deserialization and integrity.
  3. Live end-to-end inference verification.
"""

import os
import numpy as np
import joblib

def run_health_check():
    print("=" * 60)
    print("THEORIA — SYSTEM HEALTH & ARTIFACT VERIFICATION")
    print("=" * 60)

    # 1. Dataset checks
    csv_path = "consolidated_traffic_data.csv"
    assert os.path.exists(csv_path), f"Missing raw dataset: {csv_path}"
    print(f"[✓] Raw Dataset: {csv_path} ({os.path.getsize(csv_path)/(1024*1024):.1f} MB)")

    # 2. Partition checks
    npz_path = "data/prepared_data.npz"
    assert os.path.exists(npz_path), f"Missing partitions: {npz_path}"
    data = np.load(npz_path, allow_pickle=True)
    print(f"[✓] Prepared Data NPZ: {npz_path}")
    print(f"    - Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")
    print(f"    - Classes ({len(data['classes'])}): {list(data['classes'])}")

    # 3. Model checks
    models_to_check = [
        "scaler.pkl", "label_encoder.pkl", "dummy_classifier.pkl",
        "decision_tree.pkl", "k-nearest_neighbors.pkl",
        "random_forest.pkl", "gradient_boosting_(histgb).pkl",
        "multi-layer_perceptron.pkl"
    ]
    print("\n[>] Checking Trained Models (models/):")
    for m in models_to_check:
        p = os.path.join("models", m)
        assert os.path.exists(p), f"Missing model: {p}"
        obj = joblib.load(p)
        print(f"    [✓] {m:<32} ({type(obj).__name__})")

    # 4. Results checks
    plots_to_check = [
        "model_comparison.json", "model_benchmark_comparison.png",
        "theoria_nodal_relationship_graph.png",
        "random_forest_confusion_matrix.png",
        "feature_importance_ranking.png", "shap_global_bar.png",
        "shap_voip_dot.png", "shap_values.npz"
    ]
    print("\n[>] Checking Evaluation Artifacts (results/):")
    for pl in plots_to_check:
        p = os.path.join("results", pl)
        assert os.path.exists(p), f"Missing result artifact: {p}"
        print(f"    [✓] {pl:<35} ({os.path.getsize(p)/1024:.1f} KB)")

    # 5. Live inference test
    print("\n[>] Testing End-to-End Live Inference:")
    rf = joblib.load("models/random_forest.pkl")
    le = joblib.load("models/label_encoder.pkl")
    sample_x = data['X_test'][0:1]
    sample_y = data['y_app_test'][0]
    
    pred_idx = rf.predict(sample_x)[0]
    probs = rf.predict_proba(sample_x)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    true_label = le.inverse_transform([sample_y])[0]
    
    print(f"    - Ground Truth:  {true_label}")
    print(f"    - Prediction:    {pred_label} (Confidence: {probs[pred_idx]*100:.1f}%)")
    assert pred_label == true_label, "Inference verification mismatch"
    print("    [✓] Live inference verified successfully!")

    print("=" * 60)
    print("ALL CHECKS PASSED: Theoria is 100% operational!")
    print("=" * 60)

if __name__ == "__main__":
    run_health_check()
