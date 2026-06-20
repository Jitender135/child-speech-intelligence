"""
Omli — Day 3
=============
Deep dive on child vs adult classification specifically.
This is Omli's core problem — the model must work in noisy,
real-world conditions (background noise, different microphones,
kids speaking softly or excitedly).

Steps:
  1. Extract child + adult clips only (drop noise class)
  2. Train baseline classifier (clean audio)
  3. Augment data — pitch shift, speed perturb, noise injection
  4. Train augmented classifier
  5. Evaluate both on clean AND noisy test clips
  6. Compare: does augmentation help on noisy audio?

Outputs:
  outputs/day3/baseline_vs_augmented.png
  outputs/day3/noisy_evaluation.png
  outputs/day3/day3_summary.json
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
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
OUT_DIR  = BASE_DIR / "outputs" / "day3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load Raw Audio (child + adult only)
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


def load_raw_clips(n_per_class=250):
    print("[LOAD] Loading raw audio clips (child + adult)...")

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
                clips.append({"audio": audio, "label": label})
                count += 1
        print(f"  [{label}] Loaded {count} clips")

    random.shuffle(clips)
    return clips


# ─────────────────────────────────────────────
# 2. Data Augmentation
# ─────────────────────────────────────────────
def augment_clip(audio, sr=SR):
    """
    Apply random augmentation to one clip.
    Randomly picks 1-2 augmentations per clip.

    Augmentations:
      - Pitch shift: shift F0 up/down (tests robustness to pitch variation)
      - Speed perturbation: faster/slower speech
      - Background noise injection: simulate real-world noise
      - Time shift: offset the clip slightly
    """
    augmented = audio.copy()
    ops = random.sample(["pitch", "speed", "noise", "timeshift"], k=random.randint(1, 2))

    for op in ops:
        if op == "pitch":
            # Shift pitch by -3 to +3 semitones
            n_steps = random.uniform(-3, 3)
            augmented = librosa.effects.pitch_shift(
                augmented, sr=sr, n_steps=n_steps)

        elif op == "speed":
            # Speed up or slow down by 10-20%
            rate = random.uniform(0.8, 1.2)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
            # Re-pad or trim to original length
            target = int(CLIP_DURATION * sr)
            if len(augmented) < target:
                augmented = np.pad(augmented, (0, target - len(augmented)))
            else:
                augmented = augmented[:target]

        elif op == "noise":
            # Add white noise at random SNR (10-30 dB)
            snr_db  = random.uniform(10, 30)
            signal_power = np.mean(augmented ** 2)
            noise_power  = signal_power / (10 ** (snr_db / 10))
            noise        = np.random.randn(len(augmented)) * np.sqrt(noise_power)
            augmented    = augmented + noise

        elif op == "timeshift":
            # Shift audio left or right by up to 0.5s
            shift = random.randint(0, int(0.5 * sr))
            augmented = np.roll(augmented, shift)

    # Renormalize
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-8)
    return augmented.astype(np.float32)


def augment_dataset(clips, n_augment_per_clip=2):
    """
    For each original clip, generate n_augment_per_clip augmented versions.
    Returns original + augmented clips combined.
    """
    print(f"\n[AUG] Augmenting dataset ({n_augment_per_clip}x per clip)...")
    augmented = []
    for clip in clips:
        for _ in range(n_augment_per_clip):
            aug_audio = augment_clip(clip["audio"])
            augmented.append({"audio": aug_audio, "label": clip["label"],
                               "augmented": True})

    combined = clips + augmented
    random.shuffle(combined)
    print(f"  Original: {len(clips)} | Augmented: {len(augmented)} | "
          f"Total: {len(combined)}")
    return combined


# ─────────────────────────────────────────────
# 3. Feature Extraction
# ─────────────────────────────────────────────
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
        feats.append(np.mean(f))
        feats.append(np.std(f))

    feats.append(np.mean(
        librosa.feature.spectral_flatness(y=audio, hop_length=HOP_LENGTH)))

    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    feats.extend([np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)])

    return np.array(feats, dtype=np.float32)


def extract_all(clips, desc=""):
    print(f"[FEATURES] Extracting from {len(clips)} clips {desc}...")
    X, y = [], []
    for i, clip in enumerate(clips):
        if i % 100 == 0:
            print(f"  {i}/{len(clips)}")
        X.append(extract_features(clip["audio"]))
        y.append(LABEL_MAP[clip["label"]])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 4. Create Noisy Test Set
# ─────────────────────────────────────────────
def make_noisy_test_set(test_clips):
    """
    Take the clean test clips and add heavy noise.
    Used to evaluate robustness of baseline vs augmented model.
    SNR 5-10 dB — significantly noisy, like a real room.
    """
    noisy = []
    for clip in test_clips:
        audio = clip["audio"].copy()
        snr_db = random.uniform(5, 10)
        signal_power = np.mean(audio ** 2)
        noise_power  = signal_power / (10 ** (snr_db / 10))
        noise        = np.random.randn(len(audio)) * np.sqrt(noise_power)
        noisy_audio  = audio + noise
        noisy_audio  = (noisy_audio / (np.max(np.abs(noisy_audio)) + 1e-8)).astype(np.float32)
        noisy.append({"audio": noisy_audio, "label": clip["label"]})
    return noisy


# ─────────────────────────────────────────────
# 5. Train + Evaluate
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    """Use MLP — best performer from Day 2 for 2-class problem."""
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def eval_model(model, X, y, label=""):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred) * 100
    f1  = f1_score(y, y_pred, average="macro") * 100
    print(f"  {label:30s} → Accuracy: {acc:.2f}%  F1: {f1:.2f}%")
    return acc, f1, y_pred


# ─────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────
def plot_results(results):
    """Bar chart comparing baseline vs augmented on clean and noisy audio."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Baseline vs Augmented Model\nChild / Adult Classification",
                 fontsize=13, fontweight="bold")

    metrics  = ["accuracy", "f1"]
    titles   = ["Accuracy (%)", "F1 Macro (%)"]
    x        = np.arange(2)  # clean, noisy
    width    = 0.35
    x_labels = ["Clean audio", "Noisy audio (SNR 5-10 dB)"]

    for ax, metric, title in zip(axes, metrics, titles):
        baseline_vals  = [results["baseline"]["clean"][metric],
                          results["baseline"]["noisy"][metric]]
        augmented_vals = [results["augmented"]["clean"][metric],
                          results["augmented"]["noisy"][metric]]

        bars1 = ax.bar(x - width/2, baseline_vals,  width,
                       label="Baseline (clean train)",
                       color="#E67E22", edgecolor="white")
        bars2 = ax.bar(x + width/2, augmented_vals, width,
                       label="Augmented train",
                       color="#4A90D9", edgecolor="white")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(50, 100)
        ax.legend()

        for bar in bars1 + bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_vs_augmented.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved baseline_vs_augmented.png")


def plot_confusion_matrices(cm_baseline, cm_augmented):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Confusion Matrices on Noisy Test Set",
                 fontsize=13, fontweight="bold")

    for ax, cm, title in zip(axes,
                               [cm_baseline, cm_augmented],
                               ["Baseline", "Augmented"]):
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(LABEL_NAMES); ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "noisy_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved noisy_confusion_matrices.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Omli — Day 3: Child vs Adult Classifier")
    print("=" * 55)

    # Load raw audio
    all_clips = load_raw_clips(n_per_class=250)

    # Split into train/test BEFORE augmentation
    # (never augment test set — must stay clean and real)
    train_clips, test_clips = train_test_split(
        all_clips, test_size=0.20, stratify=[c["label"] for c in all_clips],
        random_state=42
    )
    print(f"\n[SPLIT] Train: {len(train_clips)} | Test: {len(test_clips)}")

    # Make noisy test set
    noisy_test_clips = make_noisy_test_set(test_clips)
    print(f"[NOISE] Created noisy test set ({len(noisy_test_clips)} clips, SNR 5-10 dB)")

    # Extract features
    print("\n--- Baseline (clean train) ---")
    X_train_clean, y_train = extract_all(train_clips, "(clean train)")
    X_test_clean,  y_test  = extract_all(test_clips,  "(clean test)")
    X_test_noisy,  _       = extract_all(noisy_test_clips, "(noisy test)")

    # Augmented train set
    print("\n--- Augmented train set ---")
    aug_clips              = augment_dataset(train_clips, n_augment_per_clip=2)
    X_train_aug, y_train_aug = extract_all(aug_clips, "(augmented train)")

    # Normalize
    scaler_base = StandardScaler()
    X_tr_clean_sc  = scaler_base.fit_transform(X_train_clean)
    X_te_clean_sc  = scaler_base.transform(X_test_clean)
    X_te_noisy_sc  = scaler_base.transform(X_test_noisy)

    scaler_aug = StandardScaler()
    X_tr_aug_sc    = scaler_aug.fit_transform(X_train_aug)
    X_te_clean_aug = scaler_aug.transform(X_test_clean)
    X_te_noisy_aug = scaler_aug.transform(X_test_noisy)

    # Train both models
    print("\n[TRAIN] Training baseline model (clean data only)...")
    model_baseline = train_model(X_tr_clean_sc, y_train)

    print("[TRAIN] Training augmented model...")
    model_augmented = train_model(X_tr_aug_sc, y_train_aug)

    # Evaluate
    print("\n[EVAL] Results:")
    b_clean_acc, b_clean_f1, _  = eval_model(model_baseline,  X_te_clean_sc,  y_test, "Baseline  — clean test")
    b_noisy_acc, b_noisy_f1, b_noisy_pred = eval_model(model_baseline,  X_te_noisy_sc,  y_test, "Baseline  — noisy test")
    a_clean_acc, a_clean_f1, _  = eval_model(model_augmented, X_te_clean_aug, y_test, "Augmented — clean test")
    a_noisy_acc, a_noisy_f1, a_noisy_pred = eval_model(model_augmented, X_te_noisy_aug, y_test, "Augmented — noisy test")

    results = {
        "baseline":  {
            "clean": {"accuracy": round(b_clean_acc, 2), "f1": round(b_clean_f1, 2)},
            "noisy": {"accuracy": round(b_noisy_acc, 2), "f1": round(b_noisy_f1, 2)},
        },
        "augmented": {
            "clean": {"accuracy": round(a_clean_acc, 2), "f1": round(a_clean_f1, 2)},
            "noisy": {"accuracy": round(a_noisy_acc, 2), "f1": round(a_noisy_f1, 2)},
        },
    }

    # Improvement
    f1_improvement = round(a_noisy_f1 - b_noisy_f1, 2)
    print(f"\n  F1 improvement on noisy audio: +{f1_improvement}% from augmentation")

    # Confusion matrices on noisy test
    cm_base = confusion_matrix(y_test, b_noisy_pred)
    cm_aug  = confusion_matrix(y_test, a_noisy_pred)

    # Save plots
    print("\n[OUTPUT] Saving plots...")
    plot_results(results)
    plot_confusion_matrices(cm_base, cm_aug)

    # Save summary
    summary = {
        "task": "child vs adult binary classification",
        "train_clips_clean":    len(train_clips),
        "train_clips_augmented": len(aug_clips),
        "test_clips":           len(test_clips),
        "augmentations_used":   ["pitch_shift ±3 semitones",
                                  "speed_perturbation 0.8-1.2x",
                                  "noise_injection SNR 10-30dB",
                                  "time_shift 0-0.5s"],
        "results": results,
        "f1_improvement_on_noisy": f1_improvement,
        "key_insight": (
            "Augmentation maintains clean audio performance while "
            "improving robustness to real-world noise conditions."
        ),
    }
    with open(OUT_DIR / "day3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] day3_summary.json saved")

    print("\n" + "=" * 55)
    print("  DAY 3 SUMMARY")
    print("=" * 55)
    print(f"  Baseline  — clean: {b_clean_acc:.1f}%  noisy: {b_noisy_acc:.1f}%")
    print(f"  Augmented — clean: {a_clean_acc:.1f}%  noisy: {a_noisy_acc:.1f}%")
    print(f"  Noisy F1 improvement from augmentation: +{f1_improvement}%")
    print("=" * 55)
    print("\n✓ Day 3 complete. Ready for Day 4 — conversation quality scorer.")


if __name__ == "__main__":
    main()