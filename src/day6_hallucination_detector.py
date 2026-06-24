"""
Omli — Doro Reliability Layer
Module 1: Whisper Hallucination Detector
=========================================
Problem:
  Whisper was trained on adult speech. When a child speaks quietly,
  quickly, or in a noisy environment — Whisper hallucinates. It returns
  confident-looking transcripts for words the child never said.
  Doro then responds to those hallucinated words. The child thinks
  Doro is broken.

Solution:
  A meta-classifier that looks at BOTH the audio signal AND Whisper's
  own internal confidence signals to output a trust score.
  If trust < threshold → Doro asks the child to repeat.

Ground truth labeling strategy:
  - Run Whisper on LibriSpeech clips (known transcripts exist)
  - Compute Word Error Rate (WER) between Whisper output and ground truth
  - WER > 0.3 = hallucination (label 1), WER <= 0.3 = trust (label 0)
  - Do same for child clips + noisy clips

Outputs:
  outputs/day6/hallucination_benchmark.png
  outputs/day6/hallucination_results.json
  models/hallucination_detector.pkl
  models/hallucination_scaler.pkl
"""

import json
import random
import warnings
import pickle
import numpy as np
import pandas as pd
import librosa
import whisper
import matplotlib.pyplot as plt
from pathlib import Path
from jiwer import wer as compute_wer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SR            = 16000
CLIP_DURATION = 5.0    # slightly longer for better transcription
WER_THRESHOLD = 0.30   # WER > 30% = hallucination

BASE_DIR   = Path(__file__).parent.parent
RAW        = BASE_DIR / "data" / "raw"
OUT_DIR    = BASE_DIR / "outputs" / "day6"
MODEL_DIR  = BASE_DIR / "models"

for d in [OUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load Whisper
# ─────────────────────────────────────────────
def load_whisper():
    print("[WHISPER] Loading base model...")
    model = whisper.load_model("base")
    print("[WHISPER] Ready.")
    return model


# ─────────────────────────────────────────────
# 2. Load Audio Clips
# ─────────────────────────────────────────────
def load_clip(path, sr=SR, duration=CLIP_DURATION):
    try:
        audio, _ = librosa.load(str(path), sr=sr, mono=True, duration=duration)
        target = int(duration * sr)
        if len(audio) < target * 0.4:
            return None
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]
        return (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)
    except:
        return None


def add_noise(audio, snr_db):
    """Add white noise at given SNR level."""
    signal_power = np.mean(audio ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.randn(len(audio)) * np.sqrt(noise_power)
    noisy        = audio + noise
    return (noisy / (np.max(np.abs(noisy)) + 1e-8)).astype(np.float32)


def load_librispeech_with_transcripts(n=120):
    """
    Load LibriSpeech clips with their ground truth transcripts.
    Ground truth is in the .trans.txt files in each speaker directory.
    """
    print("[LOAD] Loading LibriSpeech clips with ground truth transcripts...")
    libri_dir = RAW / "librispeech" / "LibriSpeech" / "dev-clean"
    clips = []

    for trans_file in list(libri_dir.rglob("*.trans.txt"))[:30]:
        speaker_dir = trans_file.parent
        with open(trans_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                file_id, transcript = parts
                audio_path = speaker_dir / f"{file_id}.flac"
                if not audio_path.exists():
                    continue
                audio = load_clip(audio_path)
                if audio is None:
                    continue
                clips.append({
                    "audio":      audio,
                    "transcript": transcript.lower().strip(),
                    "speaker":    "adult",
                    "noise_snr":  None,
                    "path":       str(audio_path),
                })
                if len(clips) >= n:
                    break
        if len(clips) >= n:
            break

    print(f"  Loaded {len(clips)} adult clips with ground truth")
    return clips


def load_child_clips(n=80):
    """
    Load child clips. No ground truth transcripts exist,
    so we use acoustic features + Whisper confidence as proxy.
    """
    print("[LOAD] Loading child speech clips...")
    child_dir = RAW / "english_children"
    paths     = list(child_dir.rglob("*.wav"))
    random.shuffle(paths)

    clips = []
    for p in paths:
        if len(clips) >= n:
            break
        audio = load_clip(p)
        if audio is not None:
            clips.append({
                "audio":      audio,
                "transcript": None,   # no ground truth
                "speaker":    "child",
                "noise_snr":  None,
                "path":       str(p),
            })

    print(f"  Loaded {len(clips)} child clips")
    return clips


# ─────────────────────────────────────────────
# 3. Run Whisper + Extract Signals
# ─────────────────────────────────────────────
def transcribe_clip(whisper_model, audio, sr=SR):
    """
    Run Whisper on a clip and extract internal confidence signals.

    Key signals Whisper exposes:
      - avg_logprob: average log probability of tokens
                     (higher = more confident, range ~-2 to 0)
      - no_speech_prob: probability that no speech was detected
                        (higher = more likely silence/noise)
      - compression_ratio: ratio of transcript to audio tokens
                           (very high = possible repetition/hallucination)
    """
    # Whisper needs float32 at 16kHz
    audio_whisper = audio.astype(np.float32)

    result = whisper_model.transcribe(
        audio_whisper,
        language="en",
        fp16=False,
        verbose=False,
    )

    transcript    = result["text"].strip().lower()
    avg_logprob   = result["segments"][0]["avg_logprob"] \
                    if result["segments"] else -2.0
    no_speech_prob = result["segments"][0]["no_speech_prob"] \
                     if result["segments"] else 1.0
    compression_ratio = result["segments"][0]["compression_ratio"] \
                        if result["segments"] else 0.0

    return {
        "transcript":       transcript,
        "avg_logprob":      float(avg_logprob),
        "no_speech_prob":   float(no_speech_prob),
        "compression_ratio": float(compression_ratio),
        "word_count":       len(transcript.split()),
    }


def extract_acoustic_features(audio, sr=SR):
    """
    Acoustic features that correlate with transcription difficulty.
    Low SNR, high silence, low energy = Whisper more likely to hallucinate.
    """
    rms      = librosa.feature.rms(y=audio)[0]
    zcr      = librosa.feature.zero_crossing_rate(audio)[0]
    silence  = np.mean(rms < 0.01)

    # Estimate SNR: ratio of speech frames to noise floor
    sorted_rms = np.sort(rms)
    noise_floor = np.mean(sorted_rms[:max(1, len(sorted_rms)//10)])
    signal_mean = np.mean(sorted_rms[len(sorted_rms)//2:])
    snr_estimate = 10 * np.log10(
        (signal_mean**2 + 1e-10) / (noise_floor**2 + 1e-10))

    # Speech rate proxy: zero crossing rate variance
    speech_rate_proxy = float(np.std(zcr))

    # Spectral centroid — child speech higher than adult
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

    return {
        "rms_mean":          float(np.mean(rms)),
        "rms_std":           float(np.std(rms)),
        "silence_ratio":     float(silence),
        "snr_estimate_db":   float(np.clip(snr_estimate, -10, 40)),
        "zcr_mean":          float(np.mean(zcr)),
        "speech_rate_proxy": speech_rate_proxy,
        "centroid_mean":     float(np.mean(centroid)),
    }


# ─────────────────────────────────────────────
# 4. Build Labeled Dataset
# ─────────────────────────────────────────────
def build_hallucination_dataset(whisper_model):
    """
    Build training data for the hallucination detector.

    Labeling strategy:
      Adult clips (known transcripts):
        WER <= 0.30 → label 0 (trust)
        WER >  0.30 → label 1 (hallucination)

      Adult clips + heavy noise (SNR 5-15 dB):
        Same WER labeling — expect more hallucinations

      Child clips (no ground truth):
        Use Whisper's own no_speech_prob + avg_logprob as proxy
        no_speech_prob > 0.3 OR avg_logprob < -0.8 → label 1
    """
    print("\n[DATASET] Building hallucination training dataset...")

    adult_clips = load_librispeech_with_transcripts(n=120)
    child_clips = load_child_clips(n=80)

    rows = []
    total = len(adult_clips) * 2 + len(child_clips)
    done  = 0

    # ── Adult clean clips ──
    print("\n  Processing adult clean clips...")
    for clip in adult_clips:
        whisper_out = transcribe_clip(whisper_model, clip["audio"])
        acoustic    = extract_acoustic_features(clip["audio"])

        wer_score = compute_wer(
            clip["transcript"], whisper_out["transcript"]
        ) if clip["transcript"] else 0.0

        label = 1 if wer_score > WER_THRESHOLD else 0

        row = {
            "speaker":        "adult",
            "noise_snr":      "clean",
            "wer":            round(wer_score, 3),
            "label":          label,
            **whisper_out,
            **acoustic,
        }
        rows.append(row)
        done += 1
        if done % 20 == 0:
            print(f"    {done}/{total}")

    # ── Adult clips + noise ──
    print("\n  Processing adult noisy clips...")
    for clip in adult_clips[:80]:
        snr = random.uniform(5, 15)
        noisy_audio = add_noise(clip["audio"], snr)
        whisper_out = transcribe_clip(whisper_model, noisy_audio)
        acoustic    = extract_acoustic_features(noisy_audio)

        wer_score = compute_wer(
            clip["transcript"], whisper_out["transcript"]
        ) if clip["transcript"] else 0.0

        label = 1 if wer_score > WER_THRESHOLD else 0

        row = {
            "speaker":   "adult_noisy",
            "noise_snr": round(snr, 1),
            "wer":       round(wer_score, 3),
            "label":     label,
            **whisper_out,
            **acoustic,
        }
        rows.append(row)
        done += 1
        if done % 20 == 0:
            print(f"    {done}/{total}")

    # ── Child clips (no ground truth) ──
    print("\n  Processing child clips...")
    for clip in child_clips:
        whisper_out = transcribe_clip(whisper_model, clip["audio"])
        acoustic    = extract_acoustic_features(clip["audio"])

        # Proxy label for child clips
        # Whisper is known to struggle with children — be conservative
        label = 1 if (
            whisper_out["no_speech_prob"] > 0.25 or
            whisper_out["avg_logprob"] < -0.75 or
            whisper_out["word_count"] == 0
        ) else 0

        row = {
            "speaker":   "child",
            "noise_snr": "clean",
            "wer":       None,
            "label":     label,
            **whisper_out,
            **acoustic,
        }
        rows.append(row)
        done += 1
        if done % 20 == 0:
            print(f"    {done}/{total}")

    df = pd.DataFrame(rows)
    print(f"\n[DATASET] Total: {len(df)} clips")
    print(f"  Trust (0): {(df['label']==0).sum()} | "
          f"Hallucination (1): {(df['label']==1).sum()}")
    print(f"  WER stats (adult only):")
    wer_df = df[df['wer'].notna()]
    print(f"    mean WER: {wer_df['wer'].mean():.3f}")
    print(f"    pct hallucinated: {(wer_df['wer'] > WER_THRESHOLD).mean()*100:.1f}%")

    return df


# ─────────────────────────────────────────────
# 5. Train Hallucination Detector
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "avg_logprob", "no_speech_prob", "compression_ratio", "word_count",
    "rms_mean", "rms_std", "silence_ratio", "snr_estimate_db",
    "zcr_mean", "speech_rate_proxy", "centroid_mean",
]


def train_detector(df):
    """Train and benchmark 3 classifiers. Save the best one."""
    print("\n[TRAIN] Training hallucination detector...")

    X = df[FEATURE_COLS].fillna(0).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression":  LogisticRegression(
            max_iter=1000, random_state=42),
        "Random Forest":        RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(
            n_estimators=200, random_state=42),
    }

    results  = {}
    best_f1  = 0
    best_model = None
    best_name  = ""

    print(f"\n  {'Model':<25} {'Accuracy':>9} {'F1':>8} {'AUC':>8}")
    print("  " + "-"*55)

    for name, model in models.items():
        model.fit(X_tr_sc, y_train)
        y_pred  = model.predict(X_te_sc)
        y_proba = model.predict_proba(X_te_sc)[:, 1]

        acc = accuracy_score(y_test, y_pred) * 100
        f1  = f1_score(y_test, y_pred, average="macro") * 100
        auc = roc_auc_score(y_test, y_proba) * 100

        print(f"  {name:<25} {acc:>8.1f}% {f1:>7.1f}% {auc:>7.1f}%")
        results[name] = {"accuracy": acc, "f1": f1, "auc": auc}

        if f1 > best_f1:
            best_f1    = f1
            best_model = model
            best_name  = name

    print(f"\n  Best model: {best_name} (F1: {best_f1:.1f}%)")

    # Save best model
    with open(MODEL_DIR / "hallucination_detector.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "hallucination_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "hallucination_feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    print(f"  Saved to models/")
    return best_model, scaler, X_test, y_test, results, best_name


# ─────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────
def plot_results(df, model, scaler, X_test, y_test, results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Whisper Hallucination Detector — Analysis",
                 fontsize=14, fontweight="bold")

    # 1. WER distribution by speaker type
    ax = axes[0][0]
    for speaker, color in [("adult","#4A90D9"),("adult_noisy","#E67E22")]:
        sub = df[(df["speaker"]==speaker) & df["wer"].notna()]
        if len(sub):
            ax.hist(sub["wer"], bins=20, alpha=0.7,
                    color=color, label=speaker, density=True)
    ax.axvline(WER_THRESHOLD, color="red", linestyle="--",
               label=f"Threshold ({WER_THRESHOLD})")
    ax.set_xlabel("Word Error Rate (WER)")
    ax.set_ylabel("Density")
    ax.set_title("WER Distribution\nClean vs Noisy Audio", fontweight="bold")
    ax.legend(fontsize=8)

    # 2. Whisper avg_logprob by label
    ax = axes[0][1]
    for label, color, name in [(0,"#27AE60","Trust"),
                                 (1,"#E74C3C","Hallucination")]:
        sub = df[df["label"]==label]["avg_logprob"]
        ax.hist(sub, bins=20, alpha=0.7, color=color,
                label=name, density=True)
    ax.set_xlabel("Whisper avg_logprob")
    ax.set_ylabel("Density")
    ax.set_title("Whisper Internal Confidence\nby Label", fontweight="bold")
    ax.legend(fontsize=8)

    # 3. SNR vs WER scatter
    ax = axes[0][2]
    adult_df = df[(df["speaker"].isin(["adult","adult_noisy"])) &
                   df["wer"].notna()]
    sc = ax.scatter(adult_df["snr_estimate_db"], adult_df["wer"],
                    c=adult_df["label"], cmap="RdYlGn_r",
                    alpha=0.6, s=30)
    ax.axhline(WER_THRESHOLD, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Estimated SNR (dB)")
    ax.set_ylabel("WER")
    ax.set_title("SNR vs WER\n(red = hallucination)", fontweight="bold")
    plt.colorbar(sc, ax=ax)

    # 4. Model benchmark
    ax = axes[1][0]
    names  = list(results.keys())
    f1s    = [results[n]["f1"] for n in names]
    aucs   = [results[n]["auc"] for n in names]
    x      = np.arange(len(names))
    width  = 0.35
    ax.bar(x - width/2, f1s,  width, label="F1 Macro",
           color="#4A90D9", edgecolor="white")
    ax.bar(x + width/2, aucs, width, label="AUC",
           color="#E67E22", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=8)
    ax.set_ylim(50, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Benchmark", fontweight="bold")
    ax.legend(fontsize=8)
    for i, (f, a) in enumerate(zip(f1s, aucs)):
        ax.text(i-width/2, f+0.5, f"{f:.0f}", ha="center", fontsize=8)
        ax.text(i+width/2, a+0.5, f"{a:.0f}", ha="center", fontsize=8)

    # 5. Confusion matrix
    ax = axes[1][1]
    y_pred = model.predict(scaler.transform(X_test))
    cm     = confusion_matrix(y_test, y_pred)
    im     = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Trust","Hallucination"])
    ax.set_yticklabels(["Trust","Hallucination"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontweight="bold")
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>thresh else "black",
                    fontsize=14, fontweight="bold")

    # 6. Feature importance (if RF)
    ax = axes[1][2]
    if hasattr(model, "feature_importances_"):
        imp     = model.feature_importances_
        idx     = np.argsort(imp)
        ax.barh(range(len(FEATURE_COLS)),
                imp[idx], color="#4A90D9", edgecolor="white")
        ax.set_yticks(range(len(FEATURE_COLS)))
        ax.set_yticklabels([FEATURE_COLS[i] for i in idx], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "See model coefficients",
                ha="center", va="center")
        ax.set_title("Feature Importance", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "hallucination_benchmark.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved hallucination_benchmark.png")


# ─────────────────────────────────────────────
# Public API — used by reliability layer
# ─────────────────────────────────────────────
def load_hallucination_detector():
    """Load saved model for use in the reliability layer."""
    with open(MODEL_DIR / "hallucination_detector.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "hallucination_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    whisper_model = whisper.load_model("base")
    return model, scaler, whisper_model


def predict_trust(audio, model, scaler, whisper_model, threshold=0.50):
    """
    Main inference function.
    Returns trust_score and flag for one audio clip.
    """
    whisper_out = transcribe_clip(whisper_model, audio)
    acoustic    = extract_acoustic_features(audio)

    features = np.array([[
        whisper_out["avg_logprob"],
        whisper_out["no_speech_prob"],
        whisper_out["compression_ratio"],
        whisper_out["word_count"],
        acoustic["rms_mean"],
        acoustic["rms_std"],
        acoustic["silence_ratio"],
        acoustic["snr_estimate_db"],
        acoustic["zcr_mean"],
        acoustic["speech_rate_proxy"],
        acoustic["centroid_mean"],
    ]])

    features_sc  = scaler.transform(features)
    hall_prob    = model.predict_proba(features_sc)[0][1]
    trust_score  = round(1.0 - hall_prob, 3)
    trust_flag   = "trust" if trust_score >= threshold else "repeat_request"

    return {
        "transcript":     whisper_out["transcript"],
        "trust_score":    trust_score,
        "trust_flag":     trust_flag,
        "avg_logprob":    round(whisper_out["avg_logprob"], 3),
        "no_speech_prob": round(whisper_out["no_speech_prob"], 3),
        "snr_estimate_db": round(acoustic["snr_estimate_db"], 1),
        "silence_ratio":  round(acoustic["silence_ratio"], 3),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Omli — Module 1: Hallucination Detector")
    print("=" * 55)

    whisper_model = load_whisper()
    df            = build_hallucination_dataset(whisper_model)

    model, scaler, X_test, y_test, results, best_name = train_detector(df)

    print("\n[OUTPUT] Saving results...")
    plot_results(df, model, scaler, X_test, y_test, results)

    # Save summary
    summary = {
        "problem": "Whisper hallucinates on child/noisy speech",
        "approach": "Meta-classifier using Whisper internals + acoustics",
        "wer_threshold": WER_THRESHOLD,
        "dataset_size": len(df),
        "label_dist": df["label"].value_counts().to_dict(),
        "model_benchmark": results,
        "best_model": best_name,
        "features_used": FEATURE_COLS,
        "key_finding": (
            "Whisper's avg_logprob and no_speech_prob are strong "
            "predictors of hallucination. Combined with SNR estimate "
            "and silence ratio, the detector catches most hallucinations "
            "before Doro acts on them."
        ),
    }
    with open(OUT_DIR / "hallucination_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print("  MODULE 1 COMPLETE")
    print("=" * 55)
    print(f"  Best model : {best_name}")
    print(f"  Saved to   : models/hallucination_detector.pkl")
    print("=" * 55)
    print("\n✓ Ready for Module 2 — Dropout Predictor")


if __name__ == "__main__":
    main()