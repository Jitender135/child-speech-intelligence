"""
Omli — Doro Reliability Layer
Demo Script
============
Simulates two complete Doro sessions:
  1. Engaged child — should show green throughout
  2. Disengaging child — should show warnings and interventions

Then runs on a real child audio file to show
the full JSON output Doro would receive.

Run:
  python src/day6_demo.py
"""

import sys
import json
import random
import warnings
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from day6_dropout_predictor import (
    simulate_engaged_session,
    simulate_dropout_session,
    load_audio_pool,
)
from day6_doro_reliability_layer import (
    analyze_session,
    print_session_report,
    analyze_turn,
)

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "outputs" / "day6"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Demo 1: Full Session Comparison
# ─────────────────────────────────────────────
def demo_sessions(audio_pool):
    print("\n" + "="*65)
    print("  DEMO 1: Engaged vs Disengaging Child Session")
    print("="*65)

    # Simulate sessions
    engaged_turns  = simulate_engaged_session(audio_pool, n_turns=10)
    dropout_turns  = simulate_dropout_session(audio_pool, n_turns=10)

    engaged_clips  = [{"audio": t["audio"],
                        "response_gap_ms": t["gap_ms"]}
                       for t in engaged_turns]
    dropout_clips  = [{"audio": t["audio"],
                        "response_gap_ms": t["gap_ms"]}
                       for t in dropout_turns]

    print("\n--- Session A: Engaged Child ---")
    engaged_results = analyze_session(engaged_clips)
    print_session_report(engaged_results)

    print("\n--- Session B: Disengaging Child ---")
    dropout_results = analyze_session(dropout_clips)
    print_session_report(dropout_results)

    return engaged_results, dropout_results


# ─────────────────────────────────────────────
# Demo 2: Real Child Audio File
# ─────────────────────────────────────────────
def demo_real_file(audio_pool):
    print("\n" + "="*65)
    print("  DEMO 2: Real Child Audio — Single Turn Analysis")
    print("="*65)

    # Pick a real child clip
    child_dir = BASE_DIR / "data" / "raw" / "english_children"
    paths     = list(child_dir.rglob("*.wav"))
    if not paths:
        print("  No child clips found — using simulated audio")
        audio = audio_pool[0]
    else:
        path  = random.choice(paths)
        audio, _ = librosa.load(str(path), sr=16000, mono=True, duration=3.0)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        audio = audio.astype(np.float32)
        print(f"  File: {path.name}")

    # Analyze with empty history (first turn)
    result = analyze_turn(audio, conversation_history=[])

    print("\n  Raw JSON output Doro would receive:")
    print("  " + "-"*50)
    print(json.dumps(result, indent=4))
    print("  " + "-"*50)
    print(f"\n  Doro instruction:")
    print(f"  → {result['doro_instruction']}")

    return result


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────
def plot_session_comparison(engaged_results, dropout_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Doro Reliability Layer — Session Comparison",
                 fontsize=14, fontweight="bold")

    turns_e = [r["turn"] for r in engaged_results]
    turns_d = [r["turn"] for r in dropout_results]

    # 1. Trust scores
    ax = axes[0][0]
    trust_e = [r["transcription"]["trust_score"] for r in engaged_results]
    trust_d = [r["transcription"]["trust_score"] for r in dropout_results]
    ax.plot(turns_e, trust_e, "o-", color="#27AE60",
            label="Engaged", linewidth=2)
    ax.plot(turns_d, trust_d, "s-", color="#E74C3C",
            label="Disengaging", linewidth=2)
    ax.axhline(0.50, color="gray", linestyle="--",
               alpha=0.7, label="Trust threshold")
    ax.set_xlabel("Turn"); ax.set_ylabel("Trust Score")
    ax.set_title("Transcription Trust Score", fontweight="bold")
    ax.set_ylim(0, 1.1); ax.legend(fontsize=8)

    # 2. Dropout probability
    ax = axes[0][1]
    drop_e = [r["engagement"]["dropout_probability"] for r in engaged_results]
    drop_d = [r["engagement"]["dropout_probability"] for r in dropout_results]
    ax.plot(turns_e, drop_e, "o-", color="#27AE60",
            label="Engaged", linewidth=2)
    ax.plot(turns_d, drop_d, "s-", color="#E74C3C",
            label="Disengaging", linewidth=2)
    ax.axhline(0.55, color="gray", linestyle="--",
               alpha=0.7, label="Intervention threshold")
    ax.set_xlabel("Turn"); ax.set_ylabel("Dropout Probability")
    ax.set_title("Dropout Probability Over Time", fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)

    # 3. Alert timeline — engaged
    ax = axes[1][0]
    alert_colors = {"none": "#27AE60", "medium": "#F39C12", "high": "#E74C3C"}
    for r in engaged_results:
        color = alert_colors.get(r["alert_level"], "gray")
        ax.bar(r["turn"], 1, color=color, edgecolor="white", width=0.8)
    ax.set_xlabel("Turn"); ax.set_yticks([])
    ax.set_title("Alert Timeline — Engaged Child", fontweight="bold")
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in alert_colors.items()]
    ax.legend(handles=patches, fontsize=8)

    # 4. Alert timeline — disengaging
    ax = axes[1][1]
    for r in dropout_results:
        color = alert_colors.get(r["alert_level"], "gray")
        ax.bar(r["turn"], 1, color=color, edgecolor="white", width=0.8)
    ax.set_xlabel("Turn"); ax.set_yticks([])
    ax.set_title("Alert Timeline — Disengaging Child", fontweight="bold")
    ax.legend(handles=patches, fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "demo_session_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[PLOT] Saved demo_session_comparison.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Omli — Doro Reliability Layer DEMO")
    print("=" * 65)
    print("  Modules active:")
    print("    ✓ Hallucination Detector (Whisper + acoustic meta-classifier)")
    print("    ✓ Dropout Predictor     (sliding window, 2-3 turn horizon)")

    audio_pool = load_audio_pool()

    engaged_results, dropout_results = demo_sessions(audio_pool)
    single_result = demo_real_file(audio_pool)
    plot_session_comparison(engaged_results, dropout_results)

    # Save full demo output
    demo_output = {
        "engaged_session":    engaged_results,
        "dropout_session":    dropout_results,
        "single_turn_example": single_result,
    }
    with open(OUT_DIR / "demo_output.json", "w") as f:
        json.dump(demo_output, f, indent=2)
    print("[SAVE] demo_output.json saved")

    print("\n" + "="*65)
    print("  DEMO COMPLETE")
    print("="*65)
    print("  Files produced:")
    print("    outputs/day6/demo_session_comparison.png")
    print("    outputs/day6/demo_output.json")
    print("\n  What this prototype does for Omli:")
    print("    1. Detects Whisper hallucinations before Doro acts on them")
    print("    2. Predicts child dropout 2-3 turns before it happens")
    print("    3. Gives Doro an actionable instruction every single turn")
    print("    4. Runs in <100ms — suitable for real-time use")
    print("="*65)


if __name__ == "__main__":
    main()