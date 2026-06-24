"""
Omli — Day 5A
==============
Error Analysis on Child vs Adult Classifier

Steps:
  1. Load best model (augmented MLP from Day 3)
  2. Run on full test set
  3. Find worst failure cases
  4. Categorize errors by type
  5. Implement one fix — confidence thresholding
  6. Show before/after F1

Outputs:
  outputs/day5/error_analysis.png
  outputs/day5/worst_failures.csv
  outputs/day5/day5_summary.json
"""

import json
import random
import warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    classification_report
)
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

LABEL_MAP   = {"child": 0, "adult": 1}
LABEL_NAMES = ["child", "adult"]
COLORS      = {"child": "#4A90D9", "adult": "#E67E22"}

BASE_DIR = Path(__file__).parent.parent
RAW      = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "outputs" / "day5"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load + Feature Extract
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


def extract_features(audio, sr=SR):
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


def load_dataset(n_per_class=250):
    print("[LOAD] Loading dataset...")
    child_paths = list((RAW / "english_children").rglob("*.wav"))
    adult_paths = list((RAW / "librispeech").rglob("*.flac"))
    random.shuffle(child_paths)
    random.shuffle(adult_paths)

    clips = []
    for paths, label in [(child_paths, "child"), (adult_paths, "adult")]:
        count = 0
        for p in paths:
            if count >= n_per_class:
                break
            audio = load_clip(p)
            if audio is not None:
                clips.append({
                    "audio": audio, "label": label,
                    "path": str(p), "label_id": LABEL_MAP[label]
                })
                count += 1
        print(f"  [{label}] {count} clips")

    random.shuffle(clips)
    return clips


def augment_clip(audio, sr=SR):
    ops = random.sample(["pitch", "speed", "noise", "timeshift"],
                        k=random.randint(1, 2))
    for op in ops:
        if op == "pitch":
            audio = librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=random.uniform(-3, 3))
        elif op == "speed":
            rate  = random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            target = int(CLIP_DURATION * sr)
            audio = np.pad(audio, (0, max(0, target - len(audio))))[:target]
        elif op == "noise":
            snr   = random.uniform(10, 30)
            power = np.mean(audio**2) / (10**(snr/10))
            audio = audio + np.random.randn(len(audio)) * np.sqrt(power)
        elif op == "timeshift":
            audio = np.roll(audio, random.randint(0, int(0.5 * sr)))
    return (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)


# ─────────────────────────────────────────────
# 2. Train Augmented Model
# ─────────────────────────────────────────────
def train_augmented_model(train_clips):
    print("\n[TRAIN] Building augmented training set...")
    augmented = []
    for clip in train_clips:
        for _ in range(2):
            aug = augment_clip(clip["audio"])
            augmented.append({"audio": aug, "label": clip["label"],
                               "label_id": LABEL_MAP[clip["label"]]})

    all_train = train_clips + augmented
    random.shuffle(all_train)

    print(f"[FEATURES] Extracting from {len(all_train)} training clips...")
    X = np.array([extract_features(c["audio"]) for c in all_train])
    y = np.array([c["label_id"] for c in all_train])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    print("[TRAIN] Training MLP...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu", max_iter=300,
        early_stopping=True, random_state=42
    )
    model.fit(X_sc, y)
    return model, scaler


# ─────────────────────────────────────────────
# 3. Error Analysis
# ─────────────────────────────────────────────
def analyze_errors(model, scaler, test_clips):
    """
    Run model on test set, find failures, categorize by error type.

    Error categories:
      - low_energy    : clip is very quiet (child whispering)
      - high_silence  : too much silence in clip
      - short_speech  : actual speech is very short
      - noisy_clip    : high background noise
      - pitch_overlap : adult with high pitch / child with low pitch
    """
    print("\n[EVAL] Running error analysis on test set...")

    X = np.array([extract_features(c["audio"]) for c in test_clips])
    y = np.array([c["label_id"] for c in test_clips])
    X_sc = scaler.transform(X)

    y_pred   = model.predict(X_sc)
    y_proba  = model.predict_proba(X_sc)

    # Find misclassified clips
    errors = []
    for i, (clip, pred, true, proba) in enumerate(
            zip(test_clips, y_pred, y, y_proba)):

        is_wrong = (pred != true)
        confidence = np.max(proba)

        # Compute diagnostic features
        audio   = clip["audio"]
        rms     = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
        zcr     = librosa.feature.zero_crossing_rate(
                    audio, hop_length=HOP_LENGTH)[0]
        silence = np.mean(rms < 0.01)

        f0, voiced, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SR
        )
        f0_mean = float(np.nanmean(f0[voiced])) if (
            voiced is not None and voiced.any()) else 0.0

        # Categorize error type
        error_type = "correct"
        if is_wrong:
            if np.mean(rms) < 0.02:
                error_type = "low_energy"
            elif silence > 0.5:
                error_type = "high_silence"
            elif f0_mean > 180 and true == 1:
                error_type = "pitch_overlap"  # adult with high pitch
            elif f0_mean < 200 and true == 0:
                error_type = "pitch_overlap"  # child with low pitch
            elif np.std(rms) > 0.1:
                error_type = "noisy_clip"
            else:
                error_type = "other"

        errors.append({
            "clip_id":        i,
            "true_label":     LABEL_NAMES[true],
            "pred_label":     LABEL_NAMES[pred],
            "confidence":     round(float(confidence), 3),
            "is_wrong":       bool(is_wrong),
            "error_type":     error_type,
            "rms_mean":       round(float(np.mean(rms)), 4),
            "silence_ratio":  round(float(silence), 3),
            "f0_mean_hz":     round(f0_mean, 1),
            "zcr_mean":       round(float(np.mean(zcr)), 4),
        })

    df        = pd.DataFrame(errors)
    wrong_df  = df[df["is_wrong"]]
    acc       = accuracy_score(y, y_pred) * 100
    f1        = f1_score(y, y_pred, average="macro") * 100

    print(f"  Accuracy: {acc:.2f}%  F1: {f1:.2f}%")
    print(f"  Total errors: {len(wrong_df)} / {len(test_clips)}")
    print(f"  Error breakdown:")
    if len(wrong_df) > 0:
        for etype, count in wrong_df["error_type"].value_counts().items():
            print(f"    {etype}: {count}")

    return df, wrong_df, y, y_pred, y_proba, acc, f1


# ─────────────────────────────────────────────
# 4. Fix — Confidence Thresholding
# ─────────────────────────────────────────────
def apply_confidence_threshold(y, y_pred, y_proba, threshold=0.80):
    """
    Fix: instead of always predicting, only predict when confidence > threshold.
    Low-confidence predictions get flagged as 'uncertain' instead of guessing.

    This reduces errors at the cost of some abstentions.
    In production: uncertain clips get re-processed or escalated.
    """
    y_pred_thresh = []
    abstained     = 0

    for pred, proba in zip(y_pred, y_proba):
        confidence = np.max(proba)
        if confidence >= threshold:
            y_pred_thresh.append(pred)
        else:
            # Abstain — use true label proxy (in production: re-process)
            y_pred_thresh.append(-1)
            abstained += 1

    y_pred_thresh = np.array(y_pred_thresh)

    # Evaluate only on non-abstained
    mask     = y_pred_thresh != -1
    y_filt   = y[mask]
    yp_filt  = y_pred_thresh[mask]

    f1_after  = f1_score(y_filt, yp_filt, average="macro") * 100
    acc_after = accuracy_score(y_filt, yp_filt) * 100

    return f1_after, acc_after, abstained, len(y)


# ─────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────
def plot_error_analysis(df, wrong_df, y, y_pred, before_f1, after_f1,
                         abstained, total):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Day 5 — Error Analysis", fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # 1. Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(y, y_pred)
    im  = ax1.imshow(cm, cmap="Blues", interpolation="nearest")
    ax1.set_title("Confusion Matrix", fontweight="bold")
    ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
    ax1.set_xticklabels(LABEL_NAMES); ax1.set_yticklabels(LABEL_NAMES)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    plt.colorbar(im, ax=ax1)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black",
                     fontsize=14, fontweight="bold")

    # 2. Error type breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    if len(wrong_df) > 0:
        etype_counts = wrong_df["error_type"].value_counts()
        colors_pie   = ["#E74C3C","#F39C12","#9B59B6","#1ABC9C","#E67E22"]
        ax2.pie(etype_counts.values, labels=etype_counts.index,
                colors=colors_pie[:len(etype_counts)],
                autopct="%1.0f%%", startangle=90)
        ax2.set_title("Error Types", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No errors!", ha="center", va="center",
                 fontsize=14, color="green")
        ax2.set_title("Error Types", fontweight="bold")

    # 3. Confidence distribution — correct vs wrong
    ax3 = fig.add_subplot(gs[0, 2])
    correct_df = df[~df["is_wrong"]]
    ax3.hist(correct_df["confidence"], bins=15, alpha=0.7,
             color="#27AE60", label="Correct", density=True)
    if len(wrong_df) > 0:
        ax3.hist(wrong_df["confidence"], bins=15, alpha=0.7,
                 color="#E74C3C", label="Wrong", density=True)
    ax3.axvline(0.80, color="black", linestyle="--", label="Threshold 0.80")
    ax3.set_xlabel("Confidence")
    ax3.set_ylabel("Density")
    ax3.set_title("Confidence Distribution", fontweight="bold")
    ax3.legend(fontsize=8)

    # 4. F1 before vs after threshold fix
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(["Before\n(all predictions)", "After\n(threshold=0.80)"],
                    [before_f1, after_f1],
                    color=["#E74C3C", "#27AE60"], edgecolor="white", width=0.5)
    ax4.set_ylim(80, 101)
    ax4.set_ylabel("F1 Macro (%)")
    ax4.set_title("Fix: Confidence Thresholding", fontweight="bold")
    for bar, val in zip(bars, [before_f1, after_f1]):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    abstain_pct = round(abstained / total * 100, 1)
    ax4.text(0.5, 0.05,
             f"Abstained on {abstained}/{total} clips ({abstain_pct}%)",
             ha="center", transform=ax4.transAxes, fontsize=9, color="gray")

    # 5. F0 distribution of errors
    ax5 = fig.add_subplot(gs[1, 1])
    if len(wrong_df) > 0 and wrong_df["f0_mean_hz"].sum() > 0:
        child_wrong = wrong_df[wrong_df["true_label"] == "child"]["f0_mean_hz"]
        adult_wrong = wrong_df[wrong_df["true_label"] == "adult"]["f0_mean_hz"]
        if len(child_wrong) > 0:
            ax5.hist(child_wrong, bins=10, alpha=0.7,
                     color="#4A90D9", label="child misclassified")
        if len(adult_wrong) > 0:
            ax5.hist(adult_wrong, bins=10, alpha=0.7,
                     color="#E67E22", label="adult misclassified")
        ax5.axvline(180, color="black", linestyle="--",
                    label="~180Hz boundary")
        ax5.set_xlabel("F0 mean (Hz)")
        ax5.set_ylabel("Count")
        ax5.set_title("F0 of Misclassified Clips", fontweight="bold")
        ax5.legend(fontsize=8)
    else:
        ax5.text(0.5, 0.5, "No F0 errors to plot",
                 ha="center", va="center", fontsize=11)
        ax5.set_title("F0 of Misclassified Clips", fontweight="bold")

    # 6. RMS energy of correct vs wrong
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(correct_df["rms_mean"], bins=15, alpha=0.7,
             color="#27AE60", label="Correct", density=True)
    if len(wrong_df) > 0:
        ax6.hist(wrong_df["rms_mean"], bins=15, alpha=0.7,
                 color="#E74C3C", label="Wrong", density=True)
    ax6.set_xlabel("RMS Energy (mean)")
    ax6.set_ylabel("Density")
    ax6.set_title("Energy: Correct vs Wrong", fontweight="bold")
    ax6.legend(fontsize=8)

    plt.savefig(OUT_DIR / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved error_analysis.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Omli — Day 5A: Error Analysis")
    print("=" * 55)

    # Load and split
    all_clips = load_dataset(n_per_class=250)
    train_clips, test_clips = train_test_split(
        all_clips, test_size=0.20,
        stratify=[c["label"] for c in all_clips], random_state=42
    )
    print(f"[SPLIT] Train: {len(train_clips)} | Test: {len(test_clips)}")

    # Train
    model, scaler = train_augmented_model(train_clips)

    # Error analysis
    df, wrong_df, y, y_pred, y_proba, acc, f1_before = analyze_errors(
        model, scaler, test_clips)

    # Fix — confidence thresholding
    f1_after, acc_after, abstained, total = apply_confidence_threshold(
        y, y_pred, y_proba, threshold=0.80)

    print(f"\n[FIX] Confidence threshold=0.80:")
    print(f"  F1 before: {f1_before:.2f}%")
    print(f"  F1 after:  {f1_after:.2f}%")
    print(f"  Abstained: {abstained}/{total} clips")

    # Save worst failures
    if len(wrong_df) > 0:
        wrong_df.drop(columns=["is_wrong"]).to_csv(
            OUT_DIR / "worst_failures.csv", index=False)
        print(f"[SAVE] worst_failures.csv ({len(wrong_df)} errors)")

    # Plots
    plot_error_analysis(df, wrong_df, y, y_pred,
                        f1_before, f1_after, abstained, total)

    # Summary
    summary = {
        "test_clips":      total,
        "errors":          int(len(wrong_df)),
        "error_rate_pct":  round(len(wrong_df) / total * 100, 1),
        "f1_before_fix":   round(f1_before, 2),
        "f1_after_fix":    round(f1_after, 2),
        "f1_improvement":  round(f1_after - f1_before, 2),
        "abstained":       abstained,
        "abstain_pct":     round(abstained / total * 100, 1),
        "error_taxonomy":  wrong_df["error_type"].value_counts().to_dict()
                           if len(wrong_df) > 0 else {},
        "key_insight": (
            "Confidence thresholding at 0.80 improves F1 by filtering "
            "uncertain predictions. In production, abstained clips would "
            "be re-processed with a slower, more accurate model."
        ),
    }
    with open(OUT_DIR / "day5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print("  DAY 5A SUMMARY")
    print("=" * 55)
    print(f"  Test clips       : {total}")
    print(f"  Errors found     : {len(wrong_df)}")
    print(f"  Error rate       : {round(len(wrong_df)/total*100,1)}%")
    print(f"  F1 before fix    : {f1_before:.2f}%")
    print(f"  F1 after fix     : {f1_after:.2f}%")
    print(f"  Abstained clips  : {abstained}/{total}")
    print("=" * 55)
    print("\n✓ Day 5A complete. Now building the prototype...")


if __name__ == "__main__":
    main()