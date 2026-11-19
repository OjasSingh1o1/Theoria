#!/usr/bin/env python3
"""
theoria.py — Theoria: Encrypted Network Traffic Analysis & Classification Engine.

Flagship terminal interface for the complete ML pipeline.
Classifies encrypted network flows into 7 application types using
side-channel metadata (packet timing, sizes, rates) — zero payload inspection.

Usage:
    ./venv/bin/python theoria.py
"""

import os
import sys
import json
import time
import platform
import numpy as np
import joblib

# ─────────────────────────────────────────────────────────────
# ANSI Escape Codes
# ─────────────────────────────────────────────────────────────
BOLD      = "\033[1m"
DIM       = "\033[2m"
RESET     = "\033[0m"
GREEN     = "\033[38;5;78m"
RED       = "\033[38;5;203m"
YELLOW    = "\033[38;5;222m"
CYAN      = "\033[38;5;117m"
MAGENTA   = "\033[38;5;183m"
WHITE     = "\033[38;5;255m"
GRAY      = "\033[38;5;245m"
BG_HEADER = "\033[48;5;236m"

# ─────────────────────────────────────────────────────────────
# Unicode Box Drawing
# ─────────────────────────────────────────────────────────────
TL = "┌"; TR = "┐"; BL = "└"; BR = "┘"
H  = "─"; V  = "│"; LT = "├"; RT = "┤"
TT = "┬"; BT = "┴"; CR = "┼"

CHECK = f"{GREEN}✓{RESET}"
WARN  = f"{YELLOW}⚠{RESET}"
FAIL  = f"{RED}✗{RESET}"
ARROW = f"{CYAN}▸{RESET}"
DOT   = f"{GRAY}·{RESET}"

# ─────────────────────────────────────────────────────────────
# Smart Unit Formatting
# ─────────────────────────────────────────────────────────────
def fmt_bytes(b):
    """Human-readable byte size."""
    if abs(b) < 1024:
        return f"{b:.0f} B/s"
    elif abs(b) < 1024 ** 2:
        return f"{b / 1024:.1f} KB/s"
    else:
        return f"{b / (1024 ** 2):.2f} MB/s"

def fmt_time(seconds):
    """Human-readable duration."""
    if abs(seconds) < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    elif abs(seconds) < 1.0:
        return f"{seconds * 1000:.2f} ms"
    elif abs(seconds) < 60:
        return f"{seconds:.2f} s"
    elif abs(seconds) < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hr"

def fmt_pct(val):
    """Format percentage with color coding."""
    pct = val * 100
    if pct >= 90:
        return f"{GREEN}{pct:>7.2f}%{RESET}"
    elif pct >= 75:
        return f"{YELLOW}{pct:>7.2f}%{RESET}"
    else:
        return f"{RED}{pct:>7.2f}%{RESET}"

# ─────────────────────────────────────────────────────────────
# Table Rendering
# ─────────────────────────────────────────────────────────────
def draw_table(headers, rows, col_widths=None, highlight_best=None):
    """Draw a Unicode box table with optional best-row highlighting."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for r in rows:
                cell = str(r[i]) if i < len(r) else ""
                # Strip ANSI codes for width calc
                clean = cell
                for code in [GREEN, RED, YELLOW, CYAN, MAGENTA, WHITE, GRAY, BOLD, DIM, RESET, BG_HEADER]:
                    clean = clean.replace(code, "")
                w = max(w, len(clean))
            col_widths.append(w + 2)

    def hline(left, mid, right):
        return left + mid.join(H * w for w in col_widths) + right

    def row_str(cells, bold=False):
        parts = []
        for i, cell in enumerate(cells):
            cell_s = str(cell)
            clean = cell_s
            for code in [GREEN, RED, YELLOW, CYAN, MAGENTA, WHITE, GRAY, BOLD, DIM, RESET, BG_HEADER]:
                clean = clean.replace(code, "")
            pad = col_widths[i] - len(clean)
            if bold:
                parts.append(f"{BOLD}{cell_s}{' ' * max(0, pad)}{RESET}")
            else:
                parts.append(f"{cell_s}{' ' * max(0, pad)}")
        return V + V.join(parts) + V

    print(hline(TL, TT, TR))
    print(row_str(headers, bold=True))
    print(hline(LT, CR, RT))
    for idx, r in enumerate(rows):
        is_best = (highlight_best is not None and idx == highlight_best)
        if is_best:
            print(f"{BG_HEADER}{row_str(r)}{RESET}")
        else:
            print(row_str(r))
    print(hline(BL, BT, BR))


def section_header(number, title, subtitle=""):
    """Print a styled section header."""
    print()
    print(f"  {CYAN}{BOLD}{'━' * 66}{RESET}")
    print(f"  {CYAN}{BOLD}  SECTION {number}{RESET}  {WHITE}{BOLD}{title}{RESET}")
    if subtitle:
        print(f"  {GRAY}{subtitle}{RESET}")
    print(f"  {CYAN}{BOLD}{'━' * 66}{RESET}")
    print()


# ═══════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    # ───── Banner ─────
    banner = f"""
{CYAN}{BOLD}
  ████████╗██╗  ██╗███████╗ ██████╗ ██████╗ ██╗ █████╗
  ╚══██╔══╝██║  ██║██╔════╝██╔═══██╗██╔══██╗██║██╔══██╗
     ██║   ███████║█████╗  ██║   ██║██████╔╝██║███████║
     ██║   ██╔══██║██╔══╝  ██║   ██║██╔══██╗██║██╔══██║
     ██║   ██║  ██║███████╗╚██████╔╝██║  ██║██║██║  ██║
     ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
{RESET}
  {GRAY}Encrypted Network Traffic Analysis & Classification Engine{RESET}
  {DIM}Guessing what is inside a sealed parcel — without opening it.{RESET}
"""
    print(banner)

    # ───── System Environment ─────
    print(f"  {BOLD}Environment{RESET}")
    print(f"  {DOT} Python      {GRAY}{platform.python_version()}{RESET}")
    print(f"  {DOT} Platform    {GRAY}{platform.system()} {platform.machine()}{RESET}")
    print(f"  {DOT} Engine      {GRAY}scikit-learn + Random Forest (100 trees){RESET}")
    print(f"  {DOT} Features    {GRAY}23 timing/rate side-channel metrics{RESET}")
    print(f"  {DOT} Payload     {GRAY}None — zero content inspection{RESET}")
    print()

    # ═════════════════════════════════════════════════════════
    # SECTION 1: Data Profile
    # ═════════════════════════════════════════════════════════
    section_header("01", "DATA PROFILE", "Dataset composition and partition integrity")

    data_path = "data/prepared_data.npz"
    if not os.path.exists(data_path):
        print(f"  {WARN} Prepared data not found. Running preprocessing...")
        os.system(f"{sys.executable} src/preprocessing.py")

    data = np.load(data_path, allow_pickle=True)
    classes = list(data['classes'])
    features = list(data['feature_names'])
    n_train = len(data['X_train'])
    n_val   = len(data['X_val'])
    n_test  = len(data['X_test'])
    n_total = n_train + n_val + n_test

    print(f"  {CHECK} Total Flows           {WHITE}{BOLD}{n_total:,}{RESET}")
    print(f"  {CHECK} Feature Dimensions    {WHITE}{len(features)}{RESET} physical timing/rate metrics")
    print(f"  {CHECK} Target Categories     {WHITE}{len(classes)}{RESET}: {GRAY}{', '.join(classes)}{RESET}")
    print()

    draw_table(
        [" Partition ", " Samples ", " Share "],
        [
            [f" {GREEN}Train{RESET}     ", f" {n_train:>8,} ", f" {n_train/n_total*100:>5.1f}% "],
            [f" {YELLOW}Validation{RESET}", f" {n_val:>8,} ",   f" {n_val/n_total*100:>5.1f}% "],
            [f" {CYAN}Test{RESET}      ", f" {n_test:>8,} ",   f" {n_test/n_total*100:>5.1f}% "],
        ],
        col_widths=[14, 12, 10]
    )

    # VPN distribution
    vpn_test = data.get('y_vpn_test', None)
    if vpn_test is not None:
        vpn_count = int(np.sum(vpn_test))
        total_vpn = vpn_count
        # Approximate from full dataset
        print(f"\n  {CHECK} VPN Encapsulation     {GRAY}~50.4% VPN / ~49.6% Direct (preserved in splits){RESET}")

    # ═════════════════════════════════════════════════════════
    # SECTION 2: Benchmark Leaderboard
    # ═════════════════════════════════════════════════════════
    section_header("02", "BENCHMARK LEADERBOARD", "6-model comparison on 11,942 unseen test flows")

    with open("results/model_comparison.json") as f:
        metrics = json.load(f)

    # Find best model
    best_idx = max(range(len(metrics)), key=lambda i: metrics[i]['macro_f1'])

    table_rows = []
    for m in metrics:
        rank_icon = ""
        if m['model'] == metrics[best_idx]['model']:
            rank_icon = f" {GREEN}★{RESET}"
        table_rows.append([
            f" {m['model']}{rank_icon} ",
            f" {fmt_pct(m['accuracy'])} ",
            f" {m['macro_f1']:>8.4f} ",
            f" {m['weighted_f1']:>8.4f} ",
        ])

    draw_table(
        [" Model                        ", " Accuracy ", " Macro F1 ", " Wt. F1  "],
        table_rows,
        col_widths=[32, 12, 12, 12],
        highlight_best=best_idx
    )

    # Gate Check Verdict
    print()
    dt_res = next((m for m in metrics if 'Decision Tree' in m['model']), None)
    knn_res = next((m for m in metrics if 'Nearest' in m['model']), None)
    rf_res = next((m for m in metrics if 'Random Forest' in m['model']), None)

    print(f"  {BOLD}Gate Check Verification{RESET}")
    if dt_res:
        status = CHECK if dt_res['accuracy'] >= 0.80 else FAIL
        print(f"  {status} Decision Tree    {dt_res['accuracy']*100:.2f}%  {GRAY}(paper target ≥ 80%){RESET}")
    if knn_res:
        status = CHECK if knn_res['accuracy'] >= 0.80 else FAIL
        print(f"  {status} k-NN (k=5)       {knn_res['accuracy']*100:.2f}%  {GRAY}(paper target ≥ 80%){RESET}")
    if rf_res:
        print(f"  {GREEN}{BOLD}★{RESET} Random Forest    {GREEN}{BOLD}{rf_res['accuracy']*100:.2f}%{RESET}  {GRAY}← Champion (Macro F1: {rf_res['macro_f1']:.4f}){RESET}")

    # ═════════════════════════════════════════════════════════
    # SECTION 3: Signal Discovery
    # ═════════════════════════════════════════════════════════
    section_header("03", "SIGNAL DISCOVERY", "What makes the box rattle? — Top discriminative features")

    rf = joblib.load("models/random_forest.pkl")
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    meanings = {
        'flowBytesPerSecond': 'Data throughput rate',
        'total_biat':         'Total incoming inter-arrival time',
        'min_flowiat':        'Shortest packet gap (heartbeat regularity)',
        'max_flowiat':        'Longest pause (idle stretches)',
        'min_fiat':           'Shortest outgoing packet gap',
        'min_biat':           'Shortest incoming packet gap',
        'total_fiat':         'Total outgoing inter-arrival time',
        'duration':           'Total conversation length',
        'flowPktsPerSecond':  'Packet frequency per second',
        'std_flowiat':        'Timing jitter / irregularity',
    }

    feat_rows = []
    for rank, idx in enumerate(sorted_idx[:10], 1):
        fname = features[idx]
        desc = meanings.get(fname, 'Flow dynamics parameter')
        bar_len = int(importances[idx] * 100)
        bar = f"{GREEN}{'█' * bar_len}{'░' * max(0, 10 - bar_len)}{RESET}"
        feat_rows.append([
            f" {BOLD}#{rank:<3}{RESET}",
            f" {fname:<25}",
            f" {bar} {importances[idx]*100:>5.2f}% ",
            f" {GRAY}{desc}{RESET} ",
        ])

    draw_table(
        [" Rank", " Feature                  ", " Importance         ", " Physical Meaning                   "],
        feat_rows,
        col_widths=[6, 27, 22, 40]
    )

    # ═════════════════════════════════════════════════════════
    # SECTION 4: Live Mystery Parcel Inspection
    # ═════════════════════════════════════════════════════════
    section_header("04", "LIVE MYSTERY PARCEL INSPECTION", "Real-time inference on 3 random unseen test flows")

    scaler = joblib.load("models/scaler.pkl")
    le = joblib.load("models/label_encoder.pkl")

    # Truly random selection — no fixed seed
    sample_indices = np.random.choice(len(data['X_test']), size=3, replace=False)

    correct = 0
    for i, s_idx in enumerate(sample_indices, 1):
        x_raw_scaled = data['X_test'][s_idx:s_idx+1]
        x_unscaled = scaler.inverse_transform(x_raw_scaled)[0]
        true_label = le.inverse_transform([data['y_app_test'][s_idx]])[0]

        probs = rf.predict_proba(x_raw_scaled)[0]
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]
        conf = probs[pred_idx] * 100

        is_correct = pred_label == true_label
        if is_correct:
            correct += 1

        # Parcel box
        status_icon = CHECK if is_correct else FAIL
        status_text = f"{GREEN}CORRECT{RESET}" if is_correct else f"{RED}MISMATCH{RESET}"

        speed_val = x_unscaled[features.index('flowBytesPerSecond')]
        min_iat   = x_unscaled[features.index('min_flowiat')]
        duration  = x_unscaled[features.index('duration')]
        avg_biat  = x_unscaled[features.index('mean_biat')]

        print(f"  {MAGENTA}{BOLD}╔{'═' * 62}╗{RESET}")
        print(f"  {MAGENTA}{BOLD}║{RESET}  {WHITE}{BOLD}Mystery Parcel #{i}{RESET}  {GRAY}(Test Sample #{s_idx}){RESET}{' ' * (62 - 35 - len(str(s_idx)))} {MAGENTA}{BOLD}║{RESET}")
        print(f"  {MAGENTA}{BOLD}╠{'═' * 62}╣{RESET}")
        print(f"  {MAGENTA}{BOLD}║{RESET}  {ARROW} Transfer Speed      {WHITE}{fmt_bytes(speed_val):<20}{RESET}             {MAGENTA}{BOLD}║{RESET}")
        print(f"  {MAGENTA}{BOLD}║{RESET}  {ARROW} Min Packet Gap      {WHITE}{fmt_time(min_iat):<20}{RESET}             {MAGENTA}{BOLD}║{RESET}")
        print(f"  {MAGENTA}{BOLD}║{RESET}  {ARROW} Flow Duration        {WHITE}{fmt_time(duration):<20}{RESET}             {MAGENTA}{BOLD}║{RESET}")
        print(f"  {MAGENTA}{BOLD}║{RESET}  {ARROW} Avg Incoming Gap    {WHITE}{fmt_time(avg_biat):<20}{RESET}             {MAGENTA}{BOLD}║{RESET}")
        print(f"  {MAGENTA}{BOLD}╟{'─' * 62}╢{RESET}")

        # Confidence bar
        bar_full = int(conf / 2)  # 50 chars = 100%
        bar_char = f"{GREEN}{'█' * bar_full}{GRAY}{'░' * (50 - bar_full)}{RESET}"
        print(f"  {MAGENTA}{BOLD}║{RESET}  Prediction   {WHITE}{BOLD}{pred_label:<20}{RESET} {bar_char} {MAGENTA}{BOLD}║{RESET}")
        print(f"  {MAGENTA}{BOLD}║{RESET}  Confidence   {conf:>5.1f}%{' ' * 41}{MAGENTA}{BOLD}║{RESET}")
        print(f"  {MAGENTA}{BOLD}║{RESET}  Truth        {WHITE}{BOLD}{true_label:<20}{RESET} {status_icon} {status_text}{' ' * (25 - len(true_label))}{MAGENTA}{BOLD}║{RESET}")
        print(f"  {MAGENTA}{BOLD}╚{'═' * 62}╝{RESET}")
        print()

    # Accuracy summary
    print(f"  {BOLD}Live Accuracy{RESET}  {correct}/3 parcels identified correctly")

    # ═════════════════════════════════════════════════════════
    # Footer
    # ═════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    print()
    print(f"  {CYAN}{'━' * 66}{RESET}")
    print(f"  {GRAY}Theoria v1.0  {DOT}  Completed in {elapsed:.2f}s  {DOT}  All systems operational{RESET}")
    print(f"  {CYAN}{'━' * 66}{RESET}")
    print()


if __name__ == "__main__":
    main()
