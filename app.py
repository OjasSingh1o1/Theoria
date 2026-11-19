"""
app.py: The Sealed Parcel Live Demonstration Dashboard.
An interactive demonstration app for Encrypted Network Traffic Analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="The Sealed Parcel | Encrypted Traffic Classifier",
    page_icon="📦",
    layout="wide"
)

# Custom header styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📦 Guessing What is Inside a Sealed Parcel</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><b>Encrypted Network Traffic Classification</b> via Side-Channel Metadata (Packet Timing, Sizes & Rates)</div>', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    rf = joblib.load("models/random_forest.pkl")
    le = joblib.load("models/label_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    data = np.load("data/prepared_data.npz", allow_pickle=True)
    with open("results/model_comparison.json") as f:
        metrics = json.load(f)
    return rf, le, scaler, data, metrics

rf, le, scaler, data, metrics = load_assets()
classes = list(data['classes'])
features = list(data['feature_names'])

# Top navigation tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Parcel Inspector",
    "🏆 Model Arena & Benchmark",
    "📊 What Makes the Box Rattle? (Explainability)",
    "📖 The Story & Philosophy"
])

# -------------------------------------------------------------
# TAB 1: Live Parcel Inspector
# -------------------------------------------------------------
with tab1:
    st.subheader("Interactive Parcel Inspection: Guess the Sealed Application")
    st.write("Pick an encrypted network conversation below. The model has **no access to payload content, IP addresses, or ports**—only timing gaps and transfer speeds.")

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        sample_idx = st.slider("Select Flow Sample ID from Test Partition", 0, len(data['X_test'])-1, 42)
    with col_ctrl2:
        if st.button("🎲 Pick Random Mystery Parcel", use_container_width=True):
            sample_idx = int(np.random.randint(0, len(data['X_test'])))
            st.session_state['sample_idx'] = sample_idx

    x_scaled = data['X_test'][sample_idx:sample_idx+1]
    y_true_idx = data['y_app_test'][sample_idx]
    true_label = le.inverse_transform([y_true_idx])[0]

    # Convert back to unscaled for human-readable display
    x_unscaled = scaler.inverse_transform(x_scaled)[0]
    sample_df = pd.Series(x_unscaled, index=features)

    st.markdown("#### 📦 Visible Clues Outside the Parcel:")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Duration", f"{sample_df['duration']:.2f} s")
    with m2:
        st.metric("Transfer Speed", f"{sample_df['flowBytesPerSecond']/1024:.1f} KB/s")
    with m3:
        st.metric("Min Packet Gap (IAT)", f"{sample_df['min_flowiat']*1000:.1f} ms")
    with m4:
        st.metric("Avg Incoming Gap (BIAT)", f"{sample_df['mean_biat']*1000:.1f} ms")

    st.divider()

    col_btn, col_result = st.columns([1, 2])
    with col_btn:
        inspect_clicked = st.button("🔮 Guess What's Inside the Parcel", type="primary", use_container_width=True)

    if inspect_clicked or True:
        probs = rf.predict_proba(x_scaled)[0]
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100

        with col_result:
            if pred_label == true_label:
                st.success(f"### 🎉 Guess: **{pred_label}** ({confidence:.1f}% confidence)")
                st.write(f"✅ **Ground Truth:** **{true_label}** (The model correctly identified the sealed parcel!)")
            else:
                st.warning(f"### ⚠️ Guess: **{pred_label}** ({confidence:.1f}% confidence)")
                st.write(f"Ground Truth was: **{true_label}**")

        st.write("#### 📊 Confidence Meter Across All 7 Applications:")
        prob_df = pd.DataFrame({
            "Application": classes,
            "Probability (%)": probs * 100
        }).sort_values(by="Probability (%)", ascending=True)

        st.bar_chart(prob_df.set_index("Application"), horizontal=True)

# -------------------------------------------------------------
# TAB 2: Model Arena & Benchmark
# -------------------------------------------------------------
with tab2:
    st.subheader("Model Comparison Leaderboard (Test Set: 11,942 Flows)")
    st.write("All models evaluated strictly on unseen test flows using both Accuracy and Macro F1.")

    metrics_df = pd.DataFrame(metrics)
    metrics_df['Accuracy (%)'] = (metrics_df['accuracy'] * 100).round(2)
    metrics_df['Macro F1'] = metrics_df['macro_f1'].round(4)
    metrics_df['Weighted F1'] = metrics_df['weighted_f1'].round(4)
    disp_df = metrics_df[['model', 'Accuracy (%)', 'Macro F1', 'Weighted F1']].rename(columns={'model': 'Model Name'})

    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    st.info("💡 **The Gate Check:** Decision Tree (90.7%) and k-NN (82.4%) were the paper's original models. Replicating >80% proved our data hygiene was correct before training the winning Random Forest (92.6%).")

    st.subheader("Normalized Confusion Matrix")
    selected_matrix_model = st.selectbox("Select model to inspect confusion matrix:", [
        "Random Forest", "Decision Tree", "k-Nearest Neighbors", "Gradient Boosting (HistGB)", "Multi-Layer Perceptron"
    ])
    clean_fname = selected_matrix_model.lower().replace(" ", "_").replace("/", "_")
    img_path = f"results/{clean_fname}_confusion_matrix.png"
    if os.path.exists(img_path):
        st.image(img_path, caption=f"Confusion Matrix - {selected_matrix_model}", use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Explainability
# -------------------------------------------------------------
with tab3:
    st.subheader("Which Features Actually Carry the Signal?")
    st.write("Proving which physical properties carry the information that allows the classifier to identify the traffic.")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("#### 1. Gini Feature Importance Ranking")
        if os.path.exists("results/feature_importance_ranking.png"):
            st.image("results/feature_importance_ranking.png", caption="Top features ranked by decision power", use_container_width=True)
    with col_exp2:
        st.markdown("#### 2. SHAP Global Feature Impact")
        if os.path.exists("results/shap_global_bar.png"):
            st.image("results/shap_global_bar.png", caption="SHAP value distribution across classes", use_container_width=True)

    st.subheader("Per-Application Signal Signatures (SHAP Dot Plots)")
    st.write("See how specific clues push the model towards or away from individual applications:")
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        st.markdown("**VOIP (Voice Call)**")
        if os.path.exists("results/shap_voip_dot.png"):
            st.image("results/shap_voip_dot.png", caption="Tiny steady packet gaps define VoIP", use_container_width=True)
    with sub_col2:
        st.markdown("**STREAMING (Video)**")
        if os.path.exists("results/shap_streaming_dot.png"):
            st.image("results/shap_streaming_dot.png", caption="High throughput drives Streaming", use_container_width=True)
    with sub_col3:
        st.markdown("**FILE TRANSFER**")
        if os.path.exists("results/shap_file_transfer_dot.png"):
            st.image("results/shap_file_transfer_dot.png", caption="Bulk one-directional bursts", use_container_width=True)

# -------------------------------------------------------------
# TAB 4: The Story & Philosophy
# -------------------------------------------------------------
with tab4:
    st.subheader("Project Philosophy: The Sealed Parcel")
    st.markdown("""
    ### The Metaphor
    A courier hands you a sealed box you may not open. You can still learn a lot:
    * **How heavy is it?** $\\rightarrow$ Packet sizes & byte volumes
    * **Does it rattle?** $\\rightarrow$ Timing variation (jitter)
    * **How often does the sender post?** $\\rightarrow$ Inter-arrival time (IAT)
    * **What time of day / how long?** $\\rightarrow$ Flow duration

    ### The Finding
    Nearly all internet traffic is encrypted now with TLS 1.3 and VPN tunnels. Deep packet inspection is dead.
    Yet, **Random Forest achieves 92.61% accuracy** distinguishing 7 distinct application types solely using statistical timing and rate side-channels.
    """)
