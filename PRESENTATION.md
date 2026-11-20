# Theoria: Encrypted Network Traffic Analysis & Classification
## Executive Slide Deck & Presentation Guide

> **Author:** Ojas  
> **Topic:** Computer Networks & Machine Learning Applications  
> **Repository:** [github.com/OjasSingh1o1/Theoria](https://github.com/OjasSingh1o1/Theoria)

---

### Slide 1: Title & The Core Problem

* **Headline:** Theoria — Guessing What is Inside a Sealed Parcel
* **Subtitle:** Encrypted Network Traffic Classification via Side-Channel Metadata
* **Core Thesis:** Deep Packet Inspection (DPI) is obsolete in modern TLS 1.3 and VPN-encapsulated network architectures. Can machine learning identify application types *without* inspecting packet payloads or server addresses?
* **Speaker Notes:**
  > "Good morning/afternoon. Today I am presenting Theoria. In modern networking, almost everything runs over TLS 1.3 or VPN tunnels. Traditional firewalls that read packet headers or cleartext payloads can no longer see what users are doing. Theoria solves this through the Sealed Parcel metaphor: inferring what is inside an encrypted conversation solely by observing external physical dimensions like packet timing, sizes, and rates."

---

### Slide 2: The "Sealed Parcel" Paradigm

* **Visual:** The Sealed Parcel Diagram (Inspection without payload access).
* **Key Comparisons:**
  * **Weight of the Parcel:** Corresponds to total bytes transferred and packet size distribution.
  * **Arrival Rhythm:** Corresponds to Forward and Backward Inter-Arrival Times (FIAT / BIAT).
  * **Internal Rattle:** Corresponds to timing jitter and the standard deviation of inter-arrival delays.
  * **Active Window:** Corresponds to duration and idle silence periods.
* **Speaker Notes:**
  > "Imagine a courier carrying a sealed, locked package. You cannot open it. But by feeling how heavy it is, listening to how it rattles when shaken, and tracking how often deliveries arrive, you can accurately distinguish shoes from books or clothes. That is exactly how Theoria operates on network streams."

---

### Slide 3: Dataset Architecture & Data Hygiene

* **Dataset:** ISCX VPN-nonVPN 2016 (59,706 Network Flows).
* **Target Categories (7 Classes):**
  * `BROWSING` (33.5%), `VOIP` (20.2%), `FILE_TRANSFER` (14.5%), `P2P` (12.4%), `CHAT` (8.9%), `MAIL` (6.4%), `STREAMING` (4.0%).
* **Zero-Leakage Policy:**
  * Stripped all IP addresses, port numbers, and MAC addresses.
  * Standardized scaling fit *strictly* on training partition to eliminate data leakage.
  * Preserved `is_vpn` flag (50.4% VPN vs 49.6% Direct).
* **Speaker Notes:**
  > "We used the benchmark CIC-VPN2016 dataset comprising nearly 60,000 flows. Crucially, we enforced a strict zero-leakage policy: no IP addresses, no port numbers, and no MAC addresses were fed into the model. The classifier cannot cheat by memorizing server addresses—it is forced to learn real protocol dynamics."

---

### Slide 4: Multi-Model Benchmark & Gate-Check

* **Evaluation Dataset:** 11,942 Unseen Test Flows.
* **Results Table:**
  * Dummy Baseline: 33.50% Accuracy | 0.0717 Macro F1
  * Decision Tree (Paper Target >80%): **90.73% Accuracy** | **0.8893 Macro F1** [PASSED]
  * k-NN Baseline (Paper Target >80%): **82.36% Accuracy** | **0.7898 Macro F1** [PASSED]
  * **Random Forest (Winner): 92.61% Accuracy** | **0.9123 Macro F1**
  * HistGradientBoosting: 90.71% Accuracy | 0.8849 Macro F1
  * Multi-Layer Perceptron (Neural Net): 71.40% Accuracy | 0.6460 Macro F1
* **Speaker Notes:**
  > "Before training complex ensembles, we executed a paper reproduction gate check. Both Decision Tree (90.7%) and k-NN (82.4%) exceeded the original paper's target of 80%, proving our data preprocessing was sound. Our champion model, Random Forest with 100 trees, reached 92.61% accuracy and a 0.9123 Macro F1 score."

---

### Slide 5: Nodal Architecture & Signal Routing

* **Visual:** Theoria Nodal Relationship Map (`results/theoria_nodal_relationship_graph.png`).
* **4 Measurement Dimensions:**
  1. *Volume & Throughput:* `flowBytesPerSecond` (9.48%), `flowPktsPerSecond`.
  2. *Client Outgoing Rhythm (FIAT):* `min_fiat` (6.03%), user typing pauses.
  3. *Server Response Rhythm (BIAT):* `total_biat` (6.41%), incoming reply pauses.
  4. *Dynamics & Jitter:* `min_flowiat` (6.31%), `max_flowiat` (6.19%), active/idle bursts.
* **Speaker Notes:**
  > "This nodal graph illustrates how 23 raw numbers cluster into 4 physical senses: Volume, Client Rhythm, Server Response, and Timing Jitter. These flow into our ensemble hub, which routes flows to their respective applications based on distinctive non-linear signatures."

---

### Slide 6: Application Signatures Explained

* **VoIP (97% Recall):** Isochronous pacing. Voice codecs send audio packets continuously every ~20ms. Near-zero variance in minimum inter-arrival times.
* **Streaming (84% Recall):** High asymmetric incoming volume spikes during buffer refills, followed by cyclic playback pauses.
* **File Transfer (94% Recall):** Maximum transmission unit (MTU) packet flooding with full bandwidth saturation.
* **Browsing (93% Recall):** Initial asset download bursts followed by prolonged cognitive idle pauses while users read.
* **Speaker Notes:**
  > "Here is why the model works so well: VoIP cannot afford buffering delays, so packets leave at a constant 20ms heartbeat. Streaming downloads chunks in bursts and goes silent while the buffer plays. File transfer floods the pipe at full speed, and Web Browsing has long idle gaps while a human reads the page."

---

### Slide 7: Live Demonstration & Terminal CLI

* **Theoria CLI (`theoria.py`):**
  * Built with senior developer terminal aesthetics: ANSI truecolor palettes, Unicode box tables, and dynamic unit formatting.
  * Live mystery parcel inspector: picks 3 random, unseen test flows on every run to demonstrate real-time inference without pre-seeded bias.
* **Streamlit Web Dashboard (`app.py`):**
  * Full interactive parcel inspector with real-time probability distribution gauges and confusion matrix viewers.
* **Speaker Notes:**
  > "We built two delivery interfaces: a Streamlit web app, and our flagship terminal engine, theoria.py. The terminal engine displays real-time telemetry, benchmark tables, and randomly selects three mystery parcels from unseen test traffic to demonstrate live, zero-payload classification."

---

### Slide 8: Security Implications & Takeaways

* **Key Finding:** Encryption alone is insufficient to protect user privacy against traffic fingerprinting.
* **Defense Countermeasures:**
  * **Traffic Morphing:** Dynamically padding packet lengths to match uniform distributions.
  * **Chaff Packet Injection:** Artificially injecting dummy packets to obscure inter-arrival timing regularities.
* **Summary:** Theoria achieves **92.61% accuracy** and proves that physical side-channel patterns reveal the nature of internet communication even across encrypted VPN tunnels.
* **Speaker Notes:**
  > "In conclusion, Theoria proves that encryption hides data, but it does not hide behavior. To achieve true privacy in next-generation networks, protocol designers must implement traffic morphing and artificial timing jitter to prevent side-channel classification. Thank you, and I welcome any questions."
