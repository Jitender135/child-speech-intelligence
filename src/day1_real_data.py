"""
Omli — Day 1 (Real Data Version)
==================================
Loads real audio from:
  - data/raw/english_children/   → child label
  - data/raw/librispeech/        → adult label
  - data/raw/esc50/              → noise label

Then: clips → features → split → EDA → save
"""

import json
import random
import warnings
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

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
N_PER_CLASS   = 300   # clips per class (balanced)

LABEL_MAP = {"child": 0, "adult": 1, "noise": 2}
COLORS    = {"child": "#4A90D9", "adult": "#E67E22", "noise": "#9B59B6"}

BASE_DIR  = Path(__file__).parent.parent
RAW       = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
OUT_DIR   = BASE_DIR / "outputs" / "eda_plots"

for d in [DATA_PROC, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Audio Loader
# ─────────────────────────────────────────────
def load_clip(path, sr=SR, duration=CLIP_DURATION):
    """
    Load an audio file, convert to mono, resample to SR.
    Returns a fixed-length numpy array or None if file is too short.
    """
    try:
        audio, _ = librosa.load(str(path), sr=sr, mono=True, duration=duration)
        target = int(duration * sr)
        if len(audio) < target * 0.5:   # skip clips shorter than 1.5s
            return None
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio.astype(np.float32)
    except Exception as e:
        return None


def load_class(paths, label, n_max, sr=SR):
    """Load up to n_max clips from a list of file paths."""
    clips = []
    random.shuffle(paths)
    for p in paths:
        if len(clips) >= n_max:
            break
        audio = load_clip(p, sr=sr)
        if audio is not None:
            clips.append({"audio": audio, "label": label, "sr": sr,
                          "source": str(p.name)})
    print(f"  [{label}] Loaded {len(clips)} clips from {len(paths)} files")
    return clips


def build_dataset():
    print("[DATA] Loading real audio datasets...")

    # Child — english_children folder
    child_dir   = RAW / "english_children"
    child_paths = list(child_dir.rglob("*.wav"))

    # Adult — LibriSpeech FLAC files
    adult_dir   = RAW / "librispeech"
    adult_paths = list(adult_dir.rglob("*.flac"))

    # Noise — ESC-50 WAV files
    noise_dir   = RAW / "esc50"
    noise_paths = list(noise_dir.rglob("*.wav"))

    print(f"  Raw file counts → child: {len(child_paths)} | "
          f"adult: {len(adult_paths)} | noise: {len(noise_paths)}")

    child_clips = load_class(child_paths, "child", N_PER_CLASS)
    adult_clips = load_class(adult_paths, "adult", N_PER_CLASS)
    noise_clips = load_class(noise_paths, "noise", N_PER_CLASS)

    clips = child_clips + adult_clips + noise_clips
    random.shuffle(clips)

    dist = Counter(c["label"] for c in clips)
    print(f"[DATA] Total clips: {len(clips)} | Distribution: {dict(dist)}")
    return clips


# ─────────────────────────────────────────────
# 2. Feature Extraction
# ─────────────────────────────────────────────
def extract_features(audio, sr=SR):
    """
    251-dim feature vector per clip:
      MFCCs + delta + delta2  (240)
      Spectral centroid/rolloff/bandwidth/flatness (7)
      ZCR + RMS  (4)
    """
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


def extract_all(clips):
    print(f"\n[FEATURES] Extracting from {len(clips)} real clips...")
    X, y, rows = [], [], []
    for i, clip in enumerate(clips):
        if i % 100 == 0:
            print(f"  {i}/{len(clips)}")
        feats = extract_features(clip["audio"], clip["sr"])
        X.append(feats)
        y.append(LABEL_MAP[clip["label"]])
        rows.append({
            "clip_id":  i,
            "label":    clip["label"],
            "label_id": LABEL_MAP[clip["label"]],
            "source":   clip["source"],
        })
    X    = np.array(X)
    y    = np.array(y)
    meta = pd.DataFrame(rows)
    print(f"[FEATURES] Done. Shape: {X.shape}")
    return X, y, meta


# ─────────────────────────────────────────────
# 3. Stratified Split
# ─────────────────────────────────────────────
def split_data(X, y, meta):
    idx = np.arange(len(X))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=42)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, stratify=y[idx_temp], random_state=42)

    split_col             = np.array(["train"] * len(X), dtype=object)
    split_col[idx_val]    = "val"
    split_col[idx_test]   = "test"
    meta                  = meta.copy()
    meta["split"]         = split_col

    print(f"\n[SPLIT] train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")
    for s in ["train", "val", "test"]:
        dist = meta[meta["split"] == s]["label"].value_counts().to_dict()
        print(f"  {s}: {dist}")
    return idx_train, idx_val, idx_test, meta


# ─────────────────────────────────────────────
# 4. EDA Plots
# ─────────────────────────────────────────────
def plot_class_distribution(meta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Real dataset — class distribution", fontsize=13, fontweight="bold")

    counts = meta["label"].value_counts()
    axes[0].bar(counts.index, counts.values,
                color=[COLORS[l] for l in counts.index], edgecolor="white")
    axes[0].set_title("Overall")
    axes[0].set_ylabel("Clips")
    for i, (lbl, cnt) in enumerate(counts.items()):
        axes[0].text(i, cnt + 1, str(cnt), ha="center")

    split_counts = meta.groupby(["split", "label"]).size().unstack(fill_value=0)
    split_counts = split_counts.reindex(["train", "val", "test"])
    split_counts.plot(kind="bar", ax=axes[1],
                      color=[COLORS[c] for c in split_counts.columns],
                      edgecolor="white", rot=0)
    axes[1].set_title("By split")
    axes[1].legend(title="Label", bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_real_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved 01_real_class_distribution.png")


def plot_feature_distributions(X, y):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Real data — feature distributions by class",
                 fontsize=13, fontweight="bold")

    feats_to_plot = [
        ("MFCC-1 mean",      0),
        ("MFCC-2 mean",      1),
        ("MFCC-3 mean",      2),
        ("ZCR mean",       242),
        ("RMS energy mean",244),
        ("Spectral centroid", 240),
    ]

    for ax, (name, idx_f) in zip(axes.flat, feats_to_plot):
        idx_f = min(idx_f, X.shape[1] - 1)
        for lbl, lid in LABEL_MAP.items():
            ax.hist(X[y == lid, idx_f], bins=25, alpha=0.65,
                    color=COLORS[lbl], label=lbl, density=True)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_real_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved 02_real_feature_distributions.png")


def plot_spectrograms(clips):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle("Real audio — mel spectrograms by class",
                 fontsize=13, fontweight="bold")

    for row, label in enumerate(LABEL_MAP):
        samples = [c for c in clips if c["label"] == label][:2]
        for col, clip in enumerate(samples):
            ax     = axes[row][col]
            mel    = librosa.feature.melspectrogram(
                        y=clip["audio"], sr=clip["sr"], n_mels=64, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            img    = librosa.display.specshow(
                        mel_db, sr=clip["sr"], hop_length=HOP_LENGTH,
                        x_axis="time", y_axis="mel", ax=ax, cmap="magma")
            ax.set_title(f"{label} — {clip['source'][:30]}", fontsize=9)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_real_spectrograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved 03_real_spectrograms.png")


def plot_mfcc_heatmap(clips):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Real audio — average MFCC heatmap by class",
                 fontsize=13, fontweight="bold")

    for ax, (label, _) in zip(axes, LABEL_MAP.items()):
        label_clips = [c for c in clips if c["label"] == label][:80]
        mfccs = [librosa.feature.mfcc(y=c["audio"], sr=c["sr"],
                                       n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
                 for c in label_clips]
        avg = np.mean(mfccs, axis=0)
        im  = ax.imshow(avg, aspect="auto", origin="lower",
                         cmap="coolwarm", interpolation="nearest")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("MFCC coefficient")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_real_mfcc_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved 04_real_mfcc_heatmap.png")


# ─────────────────────────────────────────────
# 5. Save
# ─────────────────────────────────────────────
def save_outputs(X, y, meta, idx_train, idx_val, idx_test):
    np.save(DATA_PROC / "features.npy",  X)
    np.save(DATA_PROC / "labels.npy",    y)
    np.save(DATA_PROC / "idx_train.npy", idx_train)
    np.save(DATA_PROC / "idx_val.npy",   idx_val)
    np.save(DATA_PROC / "idx_test.npy",  idx_test)
    meta.to_csv(DATA_PROC / "metadata.csv", index=False)

    summary = {
        "data_source":  "real",
        "total_clips":  int(len(X)),
        "n_features":   int(X.shape[1]),
        "label_dist":   meta["label"].value_counts().to_dict(),
        "split_sizes":  {
            "train": int(len(idx_train)),
            "val":   int(len(idx_val)),
            "test":  int(len(idx_test)),
        },
        "datasets_used": {
            "child": "Zenodo 200495 — Kennedy et al. 2016 (real children aged ~5)",
            "adult": "LibriSpeech dev-clean — Panayotov et al. 2015",
            "noise": "ESC-50 — Piczak 2015 (environmental sounds)",
        },
        "sample_rate_hz":  SR,
        "clip_duration_s": CLIP_DURATION,
        "n_mfcc":          N_MFCC,
        "clips_per_class": N_PER_CLASS,
    }

    with open(DATA_PROC / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVE] All files written to {DATA_PROC}/")
    print(json.dumps(summary, indent=2))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Omli — Day 1 (Real Data): Feature Extraction")
    print("=" * 55)

    clips                               = build_dataset()
    X, y, meta                          = extract_all(clips)
    idx_train, idx_val, idx_test, meta  = split_data(X, y, meta)

    print("\n[EDA] Generating plots...")
    plot_class_distribution(meta)
    plot_feature_distributions(X, y)
    plot_spectrograms(clips)
    plot_mfcc_heatmap(clips)

    save_outputs(X, y, meta, idx_train, idx_val, idx_test)
    print("\n✓ Day 1 (real data) complete. Ready for Day 2.")


if __name__ == "__main__":
    main()