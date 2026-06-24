"""
Omli — Day 4
=============
Conversation Quality Scorer + Latency Benchmark

Given a sequence of audio clips simulating a child-AI conversation,
score how well the conversation is going based on:
  1. Turn-taking regularity (are gaps between turns consistent?)
  2. Child engagement (is the child speaking enough?)
  3. Response latency (how fast does the AI respond?)
  4. Speech energy trend (is the child getting more/less engaged?)
  5. Silence ratio (too much silence = bad conversation)

Also benchmarks end-to-end inference latency — how long does it
take to process one clip through the full pipeline?

Outputs:
  outputs/day4/conversation_scores.png
  outputs/day4/latency_benchmark.png
  outputs/day4/day4_summary.json
"""

import json
import time
import random
import warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SR            = 16000
CLIP_DURATION = 3.0
N_MFCC        = 40
HOP_LENGTH    = 512
N_FFT         = 2048

BASE_DIR = Path(__file__).parent.parent
RAW      = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "outputs" / "day4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load Real Audio Clips
# ─────────────────────────────────────────────
def load_clip(path, sr=SR, duration=CLIP_DURATION):
    try:
        audio, _ = librosa.load(str(path), sr=sr, mono=True, duration=duration)
        target = int(duration * sr)
        if len(audio) < target * 0.5:
            return None
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]
        return (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)
    except:
        return None


def load_clips_for_simulation():
    """Load a pool of real child and adult clips to build conversations from."""
    print("[LOAD] Loading clips for conversation simulation...")

    child_paths = list((RAW / "english_children").rglob("*.wav"))
    adult_paths = list((RAW / "librispeech").rglob("*.flac"))

    random.shuffle(child_paths)
    random.shuffle(adult_paths)

    child_clips, adult_clips = [], []

    for p in child_paths[:100]:
        a = load_clip(p)
        if a is not None:
            child_clips.append(a)

    for p in adult_paths[:100]:
        a = load_clip(p)
        if a is not None:
            adult_clips.append(a)

    print(f"  Loaded {len(child_clips)} child + {len(adult_clips)} adult clips")
    return child_clips, adult_clips


# ─────────────────────────────────────────────
# 2. Simulate Conversations
# ─────────────────────────────────────────────
def simulate_conversation(child_clips, adult_clips, n_turns=10,
                           quality="good"):
    """
    Build a simulated conversation as a sequence of turns.
    Each turn has: speaker, audio, gap_before_ms, duration_ms

    quality="good"  → regular turns, child speaks ~50% of time
    quality="poor"  → long gaps, child barely speaks, uneven turns
    """
    turns = []
    for i in range(n_turns):
        is_child_turn = (i % 2 == 0)  # alternate child/adult

        if quality == "good":
            gap_ms      = random.uniform(200, 800)    # natural pause
            child_ratio = random.uniform(0.4, 0.6)    # child speaks ~half
        else:
            gap_ms      = random.uniform(1500, 4000)  # long awkward gaps
            child_ratio = random.uniform(0.05, 0.2)   # child barely speaks

        if is_child_turn:
            # Child speaks for a fraction of the clip
            audio = random.choice(child_clips).copy()
            if quality == "poor":
                # Simulate child losing interest — fade out early
                fade_point = int(len(audio) * child_ratio)
                audio[fade_point:] = 0
            speaker = "child"
        else:
            audio   = random.choice(adult_clips).copy()
            speaker = "adult"

        turns.append({
            "turn":       i + 1,
            "speaker":    speaker,
            "audio":      audio,
            "gap_ms":     gap_ms,
            "quality":    quality,
        })

    return turns


# ─────────────────────────────────────────────
# 3. Feature Extraction per Turn
# ─────────────────────────────────────────────
def extract_turn_features(audio, sr=SR):
    """Extract acoustic features from one conversation turn."""
    # RMS energy — proxy for how actively the speaker is talking
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]

    # Zero crossing rate — higher in consonant-heavy child speech
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]

    # Silence ratio — fraction of frames below energy threshold
    energy_threshold = 0.01
    silence_ratio = np.mean(rms < energy_threshold)

    # Speech duration — frames above threshold
    speech_frames = np.sum(rms >= energy_threshold)
    speech_duration_ms = (speech_frames * HOP_LENGTH / sr) * 1000

    # Spectral centroid — higher = more high-freq content (child speech)
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, hop_length=HOP_LENGTH)[0]

    return {
        "rms_mean":           float(np.mean(rms)),
        "rms_std":            float(np.std(rms)),
        "zcr_mean":           float(np.mean(zcr)),
        "silence_ratio":      float(silence_ratio),
        "speech_duration_ms": float(speech_duration_ms),
        "centroid_mean":      float(np.mean(centroid)),
    }


# ─────────────────────────────────────────────
# 4. Conversation Scorer
# ─────────────────────────────────────────────
def score_conversation(turns):
    """
    Score a conversation on 5 dimensions (0-100 each).
    Returns per-dimension scores + overall score.

    Dimensions:
      1. turn_regularity   — are gaps between turns consistent?
      2. child_engagement  — is the child speaking enough?
      3. energy_trend      — is energy stable/increasing?
      4. silence_penalty   — penalize excessive silence
      5. balance_score     — is conversation balanced child/adult?
    """
    # Extract features per turn
    turn_features = []
    for t in turns:
        feats = extract_turn_features(t["audio"])
        feats["gap_ms"]  = t["gap_ms"]
        feats["speaker"] = t["speaker"]
        feats["turn"]    = t["turn"]
        turn_features.append(feats)

    df = pd.DataFrame(turn_features)

    # 1. Turn regularity — low std in gap times = regular = good
    gap_std        = df["gap_ms"].std()
    gap_mean       = df["gap_ms"].mean()
    regularity     = max(0, 100 - (gap_std / max(gap_mean, 1)) * 100)
    regularity     = min(100, regularity)

    # 2. Child engagement — child speech duration vs total
    child_turns    = df[df["speaker"] == "child"]
    adult_turns    = df[df["speaker"] == "adult"]
    child_speech   = child_turns["speech_duration_ms"].sum()
    total_speech   = df["speech_duration_ms"].sum()
    engagement     = (child_speech / max(total_speech, 1)) * 200
    engagement     = min(100, engagement)   # 50% child = 100 score

    # 3. Energy trend — is child energy stable or increasing?
    if len(child_turns) > 1:
        energies   = child_turns["rms_mean"].values
        trend      = np.polyfit(range(len(energies)), energies, 1)[0]
        energy_sc  = min(100, max(0, 50 + trend * 5000))
    else:
        energy_sc  = 50.0

    # 4. Silence penalty — penalize high silence ratio
    avg_silence    = df["silence_ratio"].mean()
    silence_sc     = max(0, 100 - avg_silence * 150)

    # 5. Balance — ideal is ~50/50 child/adult turns
    n_child        = len(child_turns)
    n_adult        = len(adult_turns)
    balance        = 100 - abs(n_child - n_adult) / max(len(df), 1) * 100
    balance        = max(0, balance)

    # Overall — weighted average
    overall = (
        regularity  * 0.20 +
        engagement  * 0.35 +   # most important for Omli
        energy_sc   * 0.15 +
        silence_sc  * 0.20 +
        balance     * 0.10
    )

    return {
        "turn_regularity":   round(regularity, 1),
        "child_engagement":  round(engagement, 1),
        "energy_trend":      round(energy_sc, 1),
        "silence_score":     round(silence_sc, 1),
        "balance_score":     round(balance, 1),
        "overall":           round(overall, 1),
        "avg_gap_ms":        round(gap_mean, 1),
        "child_speech_pct":  round(child_speech / max(total_speech, 1) * 100, 1),
    }


# ─────────────────────────────────────────────
# 5. Latency Benchmark
# ─────────────────────────────────────────────
def extract_features_for_latency(audio, sr=SR):
    """Same pipeline as Day 1 — used for latency measurement."""
    target = int(CLIP_DURATION * sr)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    feats = []
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for m in [mfcc, delta, delta2]:
        feats.extend(np.mean(m, axis=1))
        feats.extend(np.std(m, axis=1))

    for fn in [librosa.feature.spectral_centroid,
               librosa.feature.spectral_rolloff,
               librosa.feature.spectral_bandwidth]:
        f = fn(y=audio, sr=sr, hop_length=HOP_LENGTH)
        feats.append(np.mean(f)); feats.append(np.std(f))

    feats.append(np.mean(
        librosa.feature.spectral_flatness(y=audio, hop_length=HOP_LENGTH)))
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    feats.extend([np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)])
    return np.array(feats, dtype=np.float32)


def benchmark_latency(child_clips, adult_clips, n_runs=50):
    """
    Measure end-to-end latency:
      load → feature extraction → model prediction

    This is a production ML concern — Omli needs real-time response.
    Target: < 200ms per clip for real-time feel.
    """
    print(f"\n[LATENCY] Benchmarking inference latency ({n_runs} runs)...")

    # Train a quick model for benchmarking
    all_clips  = ([(a, 0) for a in child_clips[:80]] +
                  [(a, 1) for a in adult_clips[:80]])
    random.shuffle(all_clips)
    X = np.array([extract_features_for_latency(a) for a, _ in all_clips])
    y = np.array([l for _, l in all_clips])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    model  = MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                            max_iter=200, random_state=42)
    model.fit(X_sc, y)

    # Benchmark
    test_clips = (child_clips + adult_clips)[:n_runs]
    latencies  = []

    for audio in test_clips:
        t0    = time.perf_counter()
        feats = extract_features_for_latency(audio)
        feats_sc = scaler.transform(feats.reshape(1, -1))
        _     = model.predict(feats_sc)
        t1    = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # convert to ms

    latencies = np.array(latencies)
    results   = {
        "mean_ms":   round(float(np.mean(latencies)), 2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms":    round(float(np.percentile(latencies, 95)), 2),
        "p99_ms":    round(float(np.percentile(latencies, 99)), 2),
        "min_ms":    round(float(np.min(latencies)), 2),
        "max_ms":    round(float(np.max(latencies)), 2),
        "target_ms": 200,
        "within_target_pct": round(
            float(np.mean(latencies < 200) * 100), 1),
    }

    print(f"  Mean:   {results['mean_ms']} ms")
    print(f"  Median: {results['median_ms']} ms")
    print(f"  P95:    {results['p95_ms']} ms")
    print(f"  P99:    {results['p99_ms']} ms")
    print(f"  Within 200ms target: {results['within_target_pct']}%")

    return latencies, results


# ─────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────
def plot_conversation_scores(good_score, poor_score):
    dims = ["turn_regularity", "child_engagement",
            "energy_trend", "silence_score", "balance_score"]
    labels = ["Turn\nRegularity", "Child\nEngagement",
              "Energy\nTrend", "Silence\nScore", "Balance"]

    x     = np.arange(len(dims))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Conversation Quality Scorer", fontsize=14, fontweight="bold")

    # Bar chart — dimension scores
    ax = axes[0]
    good_vals = [good_score[d] for d in dims]
    poor_vals = [poor_score[d] for d in dims]
    ax.bar(x - width/2, good_vals, width, label="Good conversation",
           color="#4A90D9", edgecolor="white")
    ax.bar(x + width/2, poor_vals, width, label="Poor conversation",
           color="#E74C3C", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (0-100)")
    ax.set_title("Dimension Scores")
    ax.legend()
    ax.axhline(y=70, color="gray", linestyle="--", alpha=0.5, label="Target")
    for i, (g, p) in enumerate(zip(good_vals, poor_vals)):
        ax.text(i - width/2, g + 1, str(g), ha="center", fontsize=8)
        ax.text(i + width/2, p + 1, str(p), ha="center", fontsize=8)

    # Overall score gauge
    ax2 = axes[1]
    categories = ["Poor\n(0-40)", "Fair\n(40-70)", "Good\n(70-100)"]
    colors_bg  = ["#E74C3C", "#F39C12", "#27AE60"]
    ax2.barh([0, 1], [40, 30], left=[0, 40], color=colors_bg[:2],
             alpha=0.3, height=0.4)
    ax2.barh([0], [30], left=[70], color=colors_bg[2], alpha=0.3, height=0.4)

    ax2.scatter([good_score["overall"]], [0],
                color="#4A90D9", s=300, zorder=5, label=f"Good conv: {good_score['overall']}")
    ax2.scatter([poor_score["overall"]], [0],
                color="#E74C3C", s=300, marker="D", zorder=5,
                label=f"Poor conv: {poor_score['overall']}")

    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel("Overall Score")
    ax2.set_title("Overall Conversation Quality")
    ax2.legend(loc="upper left")
    ax2.set_xticks([0, 20, 40, 60, 70, 80, 100])

    plt.tight_layout()
    plt.savefig(OUT_DIR / "conversation_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved conversation_scores.png")


def plot_latency(latencies, results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("End-to-End Inference Latency", fontsize=13, fontweight="bold")

    # Histogram
    ax = axes[0]
    ax.hist(latencies, bins=20, color="#4A90D9", edgecolor="white", alpha=0.85)
    ax.axvline(results["mean_ms"],   color="#E74C3C", linestyle="--",
               label=f"Mean: {results['mean_ms']}ms")
    ax.axvline(results["p95_ms"],    color="#F39C12", linestyle="--",
               label=f"P95: {results['p95_ms']}ms")
    ax.axvline(results["target_ms"], color="#27AE60", linestyle="-",
               linewidth=2, label=f"Target: {results['target_ms']}ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Latency Distribution")
    ax.legend(fontsize=9)

    # Summary table
    ax2 = axes[1]
    ax2.axis("off")
    table_data = [
        ["Metric",          "Value"],
        ["Mean",            f"{results['mean_ms']} ms"],
        ["Median",          f"{results['median_ms']} ms"],
        ["P95",             f"{results['p95_ms']} ms"],
        ["P99",             f"{results['p99_ms']} ms"],
        ["Min",             f"{results['min_ms']} ms"],
        ["Max",             f"{results['max_ms']} ms"],
        ["Within 200ms",    f"{results['within_target_pct']}%"],
    ]
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax2.set_title("Latency Summary", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "latency_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved latency_benchmark.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Omli — Day 4: Conversation Scorer + Latency")
    print("=" * 55)

    child_clips, adult_clips = load_clips_for_simulation()

    # Simulate good and poor conversations
    print("\n[SIM] Simulating conversations...")
    good_turns = simulate_conversation(
        child_clips, adult_clips, n_turns=10, quality="good")
    poor_turns = simulate_conversation(
        child_clips, adult_clips, n_turns=10, quality="poor")

    # Score them
    print("[SCORE] Scoring conversations...")
    good_score = score_conversation(good_turns)
    poor_score = score_conversation(poor_turns)

    print(f"\n  Good conversation overall: {good_score['overall']}/100")
    print(f"  Poor conversation overall: {poor_score['overall']}/100")
    print(f"\n  Good — child engagement: {good_score['child_engagement']}%  "
          f"avg gap: {good_score['avg_gap_ms']}ms")
    print(f"  Poor — child engagement: {poor_score['child_engagement']}%  "
          f"avg gap: {poor_score['avg_gap_ms']}ms")

    # Latency benchmark
    latencies, lat_results = benchmark_latency(child_clips, adult_clips)

    # Plots
    print("\n[OUTPUT] Saving plots...")
    plot_conversation_scores(good_score, poor_score)
    plot_latency(latencies, lat_results)

    # Save summary
    summary = {
        "conversation_scoring": {
            "dimensions": ["turn_regularity", "child_engagement",
                           "energy_trend", "silence_score", "balance_score"],
            "weights":    {"child_engagement": 0.35, "turn_regularity": 0.20,
                           "silence_score": 0.20, "energy_trend": 0.15,
                           "balance_score": 0.10},
            "good_conversation": good_score,
            "poor_conversation": poor_score,
            "score_gap": round(good_score["overall"] - poor_score["overall"], 1),
        },
        "latency_benchmark": lat_results,
        "key_insight": (
            f"Scorer separates good/poor conversations by "
            f"{round(good_score['overall'] - poor_score['overall'], 1)} points. "
            f"Inference runs at {lat_results['mean_ms']}ms mean latency — "
            f"{lat_results['within_target_pct']}% within 200ms real-time target."
        ),
    }

    with open(OUT_DIR / "day4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[SAVE] day4_summary.json saved")

    print("\n" + "=" * 55)
    print("  DAY 4 SUMMARY")
    print("=" * 55)
    print(f"  Good conversation score : {good_score['overall']}/100")
    print(f"  Poor conversation score : {poor_score['overall']}/100")
    print(f"  Score gap               : {round(good_score['overall'] - poor_score['overall'], 1)} points")
    print(f"  Mean inference latency  : {lat_results['mean_ms']} ms")
    print(f"  Within 200ms target     : {lat_results['within_target_pct']}%")
    print("=" * 55)
    print("\n✓ Day 4 complete. Ready for Day 5 — error analysis.")


if __name__ == "__main__":
    main()