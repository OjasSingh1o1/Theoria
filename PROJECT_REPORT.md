# Theoria: Encrypted Network Traffic Analysis & Classification via Side-Channel Metadata

**Academic Project Report**  
**Course:** Computer Networks & Machine Learning Applications  
**Author:** Ojas  
**Date:** November 2026  
**Repository:** [github.com/OjasSingh1o1/Theoria](https://github.com/OjasSingh1o1/Theoria)

---

## Executive Summary

As cryptographic standards advance toward TLS 1.3 and ubiquitous Virtual Private Network (VPN) tunneling, classical **Deep Packet Inspection (DPI)** has become largely obsolete. Network administrators and security analysts can no longer inspect packet payloads or rely on cleartext application layer headers to classify internet traffic.

This project introduces **Theoria**, an encrypted network traffic analysis and classification engine grounded in the **Sealed Parcel** metaphor: inferring the identity of a sealed container’s contents purely from external side-channel physics without opening it. By extracting 23 direction-aware statistical features from packet inter-arrival times (IAT), byte transfer rates, burst durations, and idle periods across **59,706 real-world flows (CIC-VPN2016)**, Theoria identifies 7 distinct application categories with zero payload inspection.

### Key Milestones:
1. **Zero Data Leakage:** Stripped all network identifiers (IP addresses, port numbers, MAC addresses) so classifiers learn genuine protocol dynamics rather than server memorization.
2. **Research Gate-Check Reproduction:** Replicated the original paper's baseline models past expected thresholds: Decision Tree achieved **90.73%** (target >80%) and k-Nearest Neighbors ($k=5$) achieved **82.36%** (target >80%).
3. **Ensemble Supremacy:** Random Forest achieved the winning benchmark of **92.61% accuracy** and **0.9123 Macro F1-Score** on 11,942 unseen test flows.
4. **Signal Explainability:** Using Gini Impurity and Tree SHAP, we proved that data throughput (`flowBytesPerSecond` at 9.48%) and incoming pause times (`total_biat` at 6.41%) carry the strongest discriminative signal.

---

## 1. Problem Statement & The "Sealed Parcel" Paradigm

### 1.1 The Death of Deep Packet Inspection
Historically, network gateways inspected cleartext fields—such as the HTTP `Host` header, DNS queries, or unencrypted application payloads—to enforce Quality of Service (QoS), detect malicious activity, or monitor bandwidth utilization. Modern protocols (TLS 1.3, Encrypted Client Hello, WireGuard, and IPsec) render payload inspection mathematically infeasible without private decryption keys.

### 1.2 The Sealed Parcel Analogy
Consider a courier receiving a locked parcel. Although the inspector cannot break the lock, physical properties remain observable:
* **Weight & Volume:** Corresponds to total bytes transferred and packet size distribution.
* **Drop-off Cadence:** Corresponds to Forward and Backward Packet Inter-Arrival Times (FIAT / BIAT).
* **Rattle & Internal Movement:** Corresponds to timing jitter and the standard deviation of inter-arrival delays.
* **Delivery Window:** Corresponds to active burst duration and idle silence intervals.

Theoria formalizes this paradigm into a reproducible machine learning pipeline.

---

## 2. Dataset Profile & Target Formulation

### 2.1 Raw Dataset Architecture
We utilized the benchmark **ISCX VPN-nonVPN 2016** dataset produced by the Canadian Institute for Cybersecurity (UNB). The dataset consists of **59,706 bidirectional network flows** captured across multiple scenarios:

| Category | Raw Traffic Label | Clean Application Label | Flow Count | Dataset Share |
| :--- | :--- | :--- | :---: | :---: |
| 1 | `BROWSING`, `VPN-BROWSING` | `BROWSING` | 20,000 | 33.5% |
| 2 | `VOIP`, `VPN-VOIP` | `VOIP` | 12,061 | 20.2% |
| 3 | `FT`, `VPN-FT` | `FILE_TRANSFER` | 8,679 | 14.5% |
| 4 | `P2P`, `VPN-P2P` | `P2P` | 7,415 | 12.4% |
| 5 | `CHAT`, `VPN-CHAT` | `CHAT` | 5,344 | 8.9% |
| 6 | `MAIL`, `VPN-MAIL` | `MAIL` | 3,808 | 6.4% |
| 7 | `STREAMING`, `VPN-STREAMING`| `STREAMING` | 2,399 | 4.0% |

### 2.2 Preservation of VPN Binary Indicator
While the target classification space was mapped into 7 clean application types, we isolated the encapsulation flag into a binary column `is_vpn`:
* **VPN Flows (`is_vpn = 1`):** 30,093 flows (50.4%)
* **Direct Non-VPN Flows (`is_vpn = 0`):** 29,613 flows (49.6%)

This preservation guarantees zero information loss while enabling dual-task modeling.

---

## 3. Data Hygiene & Leakage Prevention

To prevent artificial accuracy inflation, Theoria adheres to strict engineering constraints:
1. **Feature Exclusion:** Excluded all 5-tuple addressing features (Source IP, Destination IP, Source Port, Destination Port, Protocol ID) to force models to learn statistical flow shape rather than memorizing IP address clusters.
2. **Stratified Partitioning:** A 70% / 10% / 20% stratified train/validation/test split was performed across combined `(app_label + is_vpn)` strata, ensuring equal minority representation:
   * **Train:** 41,793 flows
   * **Validation:** 5,971 flows
   * **Test (Unseen):** 11,942 flows
3. **Leak-Free Scaling:** `StandardScaler` was fitted strictly on the 41,793 training samples and subsequently applied to validation and test partitions.

---

## 4. Multi-Model Benchmarking & Gate Verification

We implemented a staged evaluation framework ranging from trivial heuristics to deep ensembles:

```
┌────────────────────────────────┬────────────┬────────────┬────────────┐
│ Model                          │ Accuracy   │ Macro F1   │ Wt. F1     │
├────────────────────────────────┼────────────┼────────────┼────────────┤
│ Dummy Classifier               │   33.50%   │   0.0717   │   0.1681   │
│ Decision Tree                  │   90.73%   │   0.8893   │   0.9067   │
│ k-Nearest Neighbors (k=5)      │   82.36%   │   0.7898   │   0.8216   │
│ Random Forest (Champion)       │   92.61%   │   0.9123   │   0.9254   │
│ HistGradientBoosting           │   90.71%   │   0.8849   │   0.9059   │
│ Multi-Layer Perceptron (MLP)   │   71.40%   │   0.6460   │   0.7027   │
└────────────────────────────────┴────────────┴────────────┴────────────┘
```

### 4.1 Gate-Check Verification
The research paper established Decision Tree and k-NN baselines. By achieving **90.73%** and **82.36%** respectively, our preprocessing pipeline validated reproduction targets (>80%) before ensemble training commenced.

### 4.2 Why Random Forest Succeeded
Tabular network telemetry is dominated by orthogonal, axis-aligned decision boundaries (e.g., `flowBytesPerSecond > 50000` AND `min_flowiat < 0.02`). The Random Forest ensemble of 100 decorrelated trees naturally partitions these boundaries without suffering from the gradient vanishing issues that degraded the Multi-Layer Perceptron (71.40%).

---

## 5. Signal Discovery & Explainability

To answer *which physical properties carry the signal*, we analyzed Gini Feature Importance and Tree SHAP values:

### 5.1 Top Discriminative Side-Channels
1. **`flowBytesPerSecond` (9.48%):** Primary volume discriminator between bulk downloads (File Transfer, Streaming) and text transactions (Chat, Browsing).
2. **`total_biat` (6.41%):** Measures cumulative incoming server pause duration.
3. **`min_flowiat` (6.31%):** Detects isochronous pacing (e.g., VoIP 20ms voice packets).
4. **`max_flowiat` (6.19%):** Captures extended human reading silences in Web Browsing.
5. **`min_fiat` (6.03%):** Identifies client upload cadence.

### 5.2 Application Signature Profiles
* **VOIP (97% True Recall):** Characterized by near-zero variance in `min_flowiat`. Audio codecs must maintain continuous pacing regardless of encryption.
* **STREAMING (84% True Recall):** Characterized by high `mean_biat` volume spikes during buffer refills, followed by cyclic idle pauses during media playback.
* **FILE TRANSFER (94% True Recall):** Characterized by continuous throughput saturation with MTU-sized packets.
* **BROWSING (93% True Recall):** Characterized by high `max_idle` and `max_flowiat` values corresponding to human cognitive pauses.

---

## 6. System Deployment & Deliverables

1. **Flagship Terminal CLI (`theoria.py`):** Standalone terminal command featuring ANSI truecolor tables, dynamic unit conversions (B/s $\rightarrow$ MB/s, µs $\rightarrow$ hours), and live mystery parcel inspection.
2. **Interactive Web Dashboard (`app.py`):** 4-tab Streamlit dashboard providing interactive parcel inspection, model arena comparisons, confusion matrices, and explainability visuals.
3. **Automated Verification Suite (`src/verify_pipeline.py`):** Self-testing health suite verifying model serialization, data partition dimensions, and inference integrity.

---

## 7. Conclusion & Security Implications

Theoria proves that encryption alone does not guarantee privacy. Even when packet payloads and server identifiers are completely concealed within VPN tunnels, side-channel metadata creates an identifiable physical fingerprint.

Future countermeasures must focus on **traffic morphing** and **chaff packet injection** (dummy packet padding and deliberate jitter insertion) to eliminate inter-arrival regularity and prevent side-channel classification.
