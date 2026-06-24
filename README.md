# Child Speech Intelligence Pipeline
### Built for Omli — AI Speech Development Platform for Kids

---

## What this is

A production-oriented ML pipeline that solves real problems in child-AI voice interaction. Built specifically around the challenges Doro (Omli's AI companion) faces when talking to young children in noisy, real-world environments.

The project has two parts:

**Part 1 — Foundation Pipeline (Days 1-5)**
End-to-end child speech classifier: VAD, child/adult detection, conversation quality scoring, and error analysis — trained on real published datasets.

**Part 2 — Doro Reliability Layer (Day 6)**
Two novel modules that address unsolved problems in Doro's production pipeline.

---

## The Problem I Identified

After researching Omli's product and the state of child speech AI, I found two silent failure modes that standard pipelines don't address:

### Problem 1 — Whisper Hallucination on Child Speech
Whisper was trained predominantly on adult speech. When a child speaks quietly, quickly, or in a noisy room — Whisper returns confident-looking transcripts for words the child never said.

**Real example caught by this system:**

| Audio filename | What child said | What Whisper transcribed |
|---|---|---|
| `the frog is sneaking out of the jar.wav` | *"the frog is sneaking out of the jar"* | *"the focus is looking out the jaw"* |

Doro would respond to words the child never said. The child thinks Doro is broken.

### Problem 2 — No Early Warning for Disengagement
By the time Omli's dashboard shows a child disengaged, the session is already over. Doro keeps talking to a child who has mentally checked out 3 turns ago.

---

## The Solution — Doro Reliability Layer

A drop-in layer that sits between the child's voice and Doro's response engine. Every turn, it answers two questions in real time:

```
Child speaks
      ↓
┌─────────────────────────────────┐
│     DORO RELIABILITY LAYER      │
│                                 │
│  1. Can we trust this           │
│     transcript?                 │
│     → trust_score: 0.54         │
│     → flag: repeat_request      │
│                                 │
│  2. Is this child about         │
│     to disengage?               │
│     → dropout_prob: 0.96        │
│     → flag: intervene           │
└─────────────────────────────────┘
      ↓
Doro receives actionable JSON
      ↓
Doro responds appropriately
```

### Output every turn

```json
{
  "turn": 7,
  "latency_ms": 1113.89,
  "alert_level": "high",
  "recommended_action": "change_topic_or_simplify",
  "transcription": {
    "transcript": "the focus is looking out the jaw.",
    "trust_score": 0.54,
    "trust_flag": "repeat_request",
    "snr_db": 9.0
  },
  "engagement": {
    "dropout_probability": 0.96,
    "dropout_flag": "intervene",
    "energy_trend": "declining",
    "gap_trend": "increasing"
  },
  "doro_instruction": "Child engagement dropping. Simplify the current question or smoothly transition to a new topic."
}
```

---

## Results

### Module 1 — Hallucination Detector

| Model | F1 | AUC |
|---|---|---|
| Logistic Regression | 72.5% | 82.5% |
| Random Forest | **77.3%** | **86.1%** |
| Gradient Boosting | 75.9% | 81.5% |

**Key finding:** Whisper's internal `avg_logprob` and `no_speech_prob` combined with acoustic SNR estimate are the strongest predictors of hallucination. 67% of child/noisy clips were flagged as hallucinations — confirming the scale of the problem.

### Module 2 — Dropout Predictor

| Model | F1 | AUC |
|---|---|---|
| Logistic Regression | 98.1% | 100% |
| Random Forest | **98.9%** | **100%** |
| Gradient Boosting | 98.5% | 99.8% |

**Key finding:** Energy trend and response gap trend across a 4-turn sliding window are sufficient to predict dropout 2-3 turns before it happens. Note: high accuracy reflects simulated training data — real labeled session data would reduce this and is the natural next step.

### Part 1 — Foundation Pipeline

| Model | Test Accuracy | Test F1 |
|---|---|---|
| Logistic Regression | 97.78% | 97.78% |
| MLP Neural Network | 96.30% | 96.30% |
| Random Forest | 93.33% | 93.32% |

**Augmentation impact:** Baseline model dropped to 50% accuracy on noisy audio. Augmented model (pitch shift, speed perturbation, noise injection) held at 95%. **+61.65% F1 improvement** purely from augmentation.

---

## Datasets Used

| Dataset | Label | Size | Citation |
|---|---|---|---|
| Kennedy et al. 2016 (Zenodo 200495) | child | 671 WAV files | Real children aged ~5 |
| LibriSpeech dev-clean | adult | 2703 FLAC files | Panayotov et al. 2015 |
| ESC-50 | noise | 2000 WAV files | Piczak 2015 |

---

## Project Structure

```
child-speech-intelligence/
├── src/
│   ├── download_datasets.py           ← auto-downloads all 3 datasets
│   ├── day1_real_data.py              ← feature extraction (251 features)
│   ├── day2_train_classifiers.py      ← 3-class benchmark
│   ├── day3_child_adult_classifier.py ← augmentation + noise robustness
│   ├── day4_conversation_scorer.py    ← quality scoring + latency benchmark
│   ├── day5_error_analysis.py         ← error taxonomy + confidence fix
│   ├── day6_hallucination_detector.py ← Module 1: Whisper trust scorer
│   ├── day6_dropout_predictor.py      ← Module 2: early dropout warning
│   ├── day6_doro_reliability_layer.py ← combined API
│   └── day6_demo.py                   ← runnable demo
├── outputs/
│   ├── eda_plots/                     ← Day 1 EDA
│   ├── results/                       ← Day 2 benchmark tables
│   ├── day3/                          ← augmentation comparison
│   ├── day4/                          ← conversation scores + latency
│   ├── day5/                          ← error analysis plots
│   └── day6/                          ← reliability layer demo
├── models/                            ← saved .pkl models
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Jitender135/child-speech-intelligence
cd child-speech-intelligence

python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Download datasets (ESC-50 + LibriSpeech auto-download, ~1GB)
python src/download_datasets.py

# Run full pipeline
python src/day1_real_data.py
python src/day2_train_classifiers.py
python src/day3_child_adult_classifier.py
python src/day4_conversation_scorer.py
python src/day5_error_analysis.py

# Train and demo the Doro Reliability Layer
python src/day6_hallucination_detector.py
python src/day6_dropout_predictor.py
python src/day6_demo.py
```

---

## Feature Engineering

### Per-clip features (251 dimensions)

| Group | Features | Dim |
|---|---|---|
| MFCCs | 40 coefficients × mean + std | 80 |
| Delta MFCCs | mean + std | 80 |
| Delta-delta MFCCs | mean + std | 80 |
| Spectral centroid / rolloff / bandwidth | mean + std | 6 |
| Spectral flatness | mean | 1 |
| Zero-crossing rate | mean + std | 2 |
| RMS energy | mean + std | 2 |
| **Total** | | **251** |

### Hallucination detector features (11 dimensions)

| Source | Feature | Why it matters |
|---|---|---|
| Whisper | avg_logprob | Low = Whisper itself is uncertain |
| Whisper | no_speech_prob | High = Whisper thinks nobody spoke |
| Whisper | compression_ratio | High = possible repetition loop |
| Audio | snr_estimate_db | Low SNR = harder to transcribe |
| Audio | silence_ratio | High silence = nothing to transcribe |
| Audio | rms_mean / std | Low energy = quiet child = harder |

### Dropout predictor features (32 dimensions)

For each of 8 acoustic signals, across a 4-turn window:
`current value`, `window mean`, `trend (slope)`, `rate of change`

Trend features are the most important — declining energy slope
and increasing gap trend are the strongest dropout predictors.

---

## Limitations and Next Steps

**Hallucination detector:**
- 77% F1 is limited by dataset size and proxy labeling for child clips
- Next step: collect 500+ child clips with human-verified transcripts
- Expected improvement: F1 → 85%+ with real labeled data

**Dropout predictor:**
- 98.9% F1 reflects simulated training data
- Next step: label real Doro sessions for ground truth dropout points
- Expected real-world F1: 75-85% — still actionable for intervention

**Latency:**
- ~4.7s per turn on CPU (dominated by Whisper)
- With GPU or Whisper `tiny` model: <200ms
- Production path: run Whisper async, reliability layer adds <50ms

---

## Key Insight for Omli

The hallucination detector already caught a real failure in the wild:

> Audio: *"the frog is sneaking out of the jar"*
> Whisper: *"the focus is looking out the jaw"*
> Trust score: 0.54 → flagged as uncertain

This is exactly the kind of silent failure that makes children think
Doro is broken — and it's happening on every low-SNR child utterance.
The fix is a lightweight meta-classifier, not a bigger model.

---

*Built as part of Omli internship project — June 2026*
*Datasets: Kennedy et al. 2016, LibriSpeech (Panayotov et al. 2015), ESC-50 (Piczak 2015)*