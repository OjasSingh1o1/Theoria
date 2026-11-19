"""
Theoria — explainability.py
Encrypted Network Traffic Analysis & Classification Engine.

Feature Importance & SHAP Explainability Analysis.
Answers the core question: "Which features actually carry the signal?"
- Gini Feature Importance from Random Forest (full test set).
- SHAP values on a stratified 2,000-sample subset of the test set.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.utils import resample

def load_data(npz_path="data/prepared_data.npz"):
    data = np.load(npz_path, allow_pickle=True)
    return (
        data['X_test'], data['y_app_test'],
        data['classes'], data['feature_names']
    )

def stratified_subsample(X, y, n=2000, random_state=42):
    """Draw a stratified subsample of n rows from X, y."""
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    class_counts = np.bincount(y)
    total = len(y)

    sampled_idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        # Allocate proportional share, at least 1
        c_share = max(1, int(round(n * len(c_idx) / total)))
        chosen = rng.choice(c_idx, size=min(c_share, len(c_idx)), replace=False)
        sampled_idx.append(chosen)

    sampled_idx = np.concatenate(sampled_idx)
    # Trim or pad to exactly n
    if len(sampled_idx) > n:
        sampled_idx = rng.choice(sampled_idx, size=n, replace=False)

    return X[sampled_idx], y[sampled_idx]

def plot_feature_importance(importances, feature_names, output_dir="results"):
    """Publication-grade bar chart of Gini feature importances."""
    os.makedirs(output_dir, exist_ok=True)

    sorted_idx = np.argsort(importances)[::-1][:15]
    top_names = feature_names[sorted_idx]
    top_vals = importances[sorted_idx] * 100

    meanings = {
        'flowBytesPerSecond': 'Throughput Speed',
        'total_biat': 'Incoming Response Pause',
        'min_flowiat': 'Fastest Packet Heartbeat',
        'max_flowiat': 'Longest Idle Interval',
        'min_fiat': 'Fastest Client Outgoing Gap',
        'min_biat': 'Fastest Server Incoming Gap',
        'total_fiat': 'Total Client Upload Silence',
        'duration': 'Conversation Length',
        'flowPktsPerSecond': 'Packet Frequency Rate',
        'std_flowiat': 'Timing Jitter / Irregularity',
        'mean_flowiat': 'Average Inter-Arrival Delay',
        'mean_biat': 'Average Server Response Delay',
        'max_biat': 'Longest Server Response Silence',
        'max_fiat': 'Longest Client Upload Silence',
        'mean_fiat': 'Average Client Outgoing Delay'
    }

    display_labels = [f"{name} ({meanings.get(name, 'Flow Dynamic')})" for name in top_names]

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAFA')

    colors = plt.cm.GnBu_r(np.linspace(0.15, 0.75, len(top_vals)))
    y_pos = np.arange(len(top_vals))
    ax.barh(y_pos, top_vals[::-1], color=colors[::-1], height=0.68, edgecolor='none', zorder=3)

    for i, v in enumerate(top_vals[::-1]):
        ax.text(v + 0.15, i, f"{v:.2f}%", va='center', ha='left', fontsize=9, fontweight='bold', color='#1E293B')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels[::-1], fontsize=9.5, color='#0F172A')
    ax.set_xlabel('Gini Feature Importance (%)', fontsize=11, fontweight='bold', color='#334155', labelpad=10)
    ax.set_title('Theoria: Top 15 Most Discriminative Side-Channel Features\nRandom Forest Importance (What Makes the Box Rattle?)',
                 fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    ax.set_xlim(0, max(top_vals) * 1.15)
    ax.grid(axis='x', linestyle=':', alpha=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_ranking.png"), dpi=300)
    plt.close()
    print(f"[+] Saved feature importance chart to results/feature_importance_ranking.png")

def plot_shap_summary(shap_values, X_subset, feature_names, class_names, output_dir="results"):
    """SHAP beeswarm summary plot."""
    os.makedirs(output_dir, exist_ok=True)

    # Global summary (bar)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_subset, feature_names=list(feature_names),
                      class_names=list(class_names), plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Feature Importance (Top 15)\nAveraged Across All Application Classes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_global_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved SHAP global bar chart to results/shap_global_bar.png")

    # Per-class dot plots for the top 3 most interesting classes
    interesting = {"VOIP": "Tiny steady gaps between packets",
                   "STREAMING": "High incoming data rates",
                   "FILE_TRANSFER": "Bulk one-directional bursts"}

    for cls_name, description in interesting.items():
        if cls_name in list(class_names):
            cls_idx = list(class_names).index(cls_name)
            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values[:, :, cls_idx], X_subset,
                              feature_names=list(feature_names), show=False, max_display=10)
            plt.title(f"SHAP Values for {cls_name}\n({description})")
            plt.tight_layout()
            fname = f"shap_{cls_name.lower()}_dot.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[+] Saved SHAP dot plot for {cls_name} to results/{fname}")

def main():
    print("[+] Loading test data and trained Random Forest model...")
    X_test, y_test, classes, feature_names = load_data()
    rf = joblib.load("models/random_forest.pkl")

    # ---------------------------------------------------------------
    # 1. Gini Feature Importance (from the full Random Forest)
    # ---------------------------------------------------------------
    print("\n[>] Computing Gini Feature Importances...")
    importances = rf.feature_importances_

    print("\nFeature Importance Ranking (Top to Bottom):")
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank:>2}. {feature_names[idx]:<25} {importances[idx]:.4f}")

    plot_feature_importance(importances, feature_names)

    # ---------------------------------------------------------------
    # 2. SHAP on a stratified 2,000-sample subset
    # ---------------------------------------------------------------
    print(f"\n[>] Drawing stratified 2,000-sample subset from test set ({len(X_test):,} total)...")
    X_sub, y_sub = stratified_subsample(X_test, y_test, n=2000)
    print(f"[+] Subset drawn: {len(X_sub)} samples")

    # Verify class representation in subset
    unique, counts = np.unique(y_sub, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {classes[u]}: {c}")

    print("\n[>] Computing SHAP values (TreeExplainer on 2,000 samples)...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sub)

    # shap_values shape: (n_samples, n_features, n_classes) for multi-class
    shap_values = np.array(shap_values)
    if shap_values.shape[0] == len(classes):
        # Some SHAP versions return (n_classes, n_samples, n_features)
        shap_values = np.transpose(shap_values, (1, 2, 0))

    print(f"[+] SHAP values computed. Shape: {shap_values.shape}")

    plot_shap_summary(shap_values, X_sub, feature_names, classes)

    # Save raw SHAP values for future analysis
    np.savez_compressed("results/shap_values.npz",
                        shap_values=shap_values,
                        X_subset=X_sub,
                        y_subset=y_sub,
                        classes=classes,
                        feature_names=feature_names)
    print("[+] Saved raw SHAP values to results/shap_values.npz")

    print("\n[✓] Phase 4 complete. All explainability artifacts saved to results/")

if __name__ == "__main__":
    main()
