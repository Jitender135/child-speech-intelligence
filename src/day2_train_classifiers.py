"""
Omli — Day 2
=============
Train and benchmark 3 classifiers on real audio features:
  1. Logistic Regression  (interpretable baseline)
  2. Random Forest        (robust, handles non-linearity)
  3. MLP Neural Network   (most powerful)

Loads from:
  data/processed/features.npy
  data/processed/labels.npy
  data/processed/idx_train.npy
  data/processed/idx_val.npy
  data/processed/idx_test.npy

Outputs:
  outputs/results/benchmark_results.csv
  outputs/results/benchmark_summary.json
  outputs/confusion_matrices/  (3 PNG plots)
  outputs/results/classification_reports.txt
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
LABEL_NAMES = ["child", "adult", "noise"]
COLORS      = {"child": "#4A90D9", "adult": "#E67E22", "noise": "#9B59B6"}

BASE_DIR  = Path(__file__).parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"
OUT_RES   = BASE_DIR / "outputs" / "results"
OUT_CM    = BASE_DIR / "outputs" / "confusion_matrices"

for d in [OUT_RES, OUT_CM]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
def load_data():
    print("[LOAD] Reading features from data/processed/...")
    X    = np.load(DATA_PROC / "features.npy")
    y    = np.load(DATA_PROC / "labels.npy")
    i_tr = np.load(DATA_PROC / "idx_train.npy")
    i_va = np.load(DATA_PROC / "idx_val.npy")
    i_te = np.load(DATA_PROC / "idx_test.npy")

    X_train, y_train = X[i_tr], y[i_tr]
    X_val,   y_val   = X[i_va], y[i_va]
    X_test,  y_test  = X[i_te], y[i_te]

    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────
# 2. Normalize Features
# ─────────────────────────────────────────────
def normalize(X_train, X_val, X_test):
    """
    StandardScaler: zero mean, unit variance.
    Fit ONLY on train — never on val/test (prevents data leakage).
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print("[NORM] Features normalized (fit on train only — no leakage)")
    return X_train, X_val, X_test, scaler


# ─────────────────────────────────────────────
# 3. Model Definitions
# ─────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
    }


# ─────────────────────────────────────────────
# 4. Train + Evaluate
# ─────────────────────────────────────────────
def evaluate(model, X_val, y_val, X_test, y_test, name):
    """Evaluate on val and test. Returns dict of metrics."""
    results = {}
    for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X_s)
        results[split_name] = {
            "accuracy":  round(accuracy_score(y_s, y_pred) * 100, 2),
            "f1_macro":  round(f1_score(y_s, y_pred, average="macro") * 100, 2),
            "precision": round(precision_score(y_s, y_pred, average="macro") * 100, 2),
            "recall":    round(recall_score(y_s, y_pred, average="macro") * 100, 2),
            "y_pred":    y_pred,
            "report":    classification_report(y_s, y_pred, target_names=LABEL_NAMES),
            "cm":        confusion_matrix(y_s, y_pred),
        }
    return results


def train_all(X_train, y_train, X_val, y_val, X_test, y_test):
    models  = get_models()
    all_res = {}

    for name, model in models.items():
        print(f"\n[TRAIN] {name}...")
        model.fit(X_train, y_train)
        print(f"  Training done.")

        res = evaluate(model, X_val, y_val, X_test, y_test, name)
        all_res[name] = {"model": model, "results": res}

        print(f"  Val  → Accuracy: {res['val']['accuracy']}%  "
              f"F1: {res['val']['f1_macro']}%")
        print(f"  Test → Accuracy: {res['test']['accuracy']}%  "
              f"F1: {res['test']['f1_macro']}%")

    return all_res


# ─────────────────────────────────────────────
# 5. Confusion Matrix Plots
# ─────────────────────────────────────────────
def plot_confusion_matrices(all_res):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")

    for ax, (name, data) in zip(axes, all_res.items()):
        cm   = data["results"]["test"]["cm"]
        f1   = data["results"]["test"]["f1_macro"]
        acc  = data["results"]["test"]["accuracy"]

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{name}\nAcc: {acc}%  F1: {f1}%", fontsize=10)
        ax.set_xticks(range(len(LABEL_NAMES)))
        ax.set_yticks(range(len(LABEL_NAMES)))
        ax.set_xticklabels(LABEL_NAMES, rotation=45)
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plt.colorbar(im, ax=ax)

        # Annotate cells
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = OUT_CM / "confusion_matrices_test.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Saved confusion_matrices_test.png")


# ─────────────────────────────────────────────
# 6. Benchmark Table
# ─────────────────────────────────────────────
def save_benchmark(all_res):
    rows = []
    for name, data in all_res.items():
        for split in ["val", "test"]:
            r = data["results"][split]
            rows.append({
                "model":     name,
                "split":     split,
                "accuracy":  r["accuracy"],
                "f1_macro":  r["f1_macro"],
                "precision": r["precision"],
                "recall":    r["recall"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES / "benchmark_results.csv", index=False)

    # Pretty print the test results
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS — TEST SET")
    print("=" * 60)
    test_df = df[df["split"] == "test"][
        ["model", "accuracy", "f1_macro", "precision", "recall"]
    ].reset_index(drop=True)
    print(test_df.to_string(index=False))
    print("=" * 60)

    # Save summary JSON
    best_model = test_df.loc[test_df["f1_macro"].idxmax(), "model"]
    summary = {
        "best_model":    best_model,
        "test_results":  test_df.to_dict(orient="records"),
        "notes": (
            "StandardScaler applied to all features. "
            "Scaler fit on train set only to prevent data leakage."
        ),
    }
    with open(OUT_RES / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] Best model on test set: {best_model}")

    return df


def save_classification_reports(all_res):
    path = OUT_RES / "classification_reports.txt"
    with open(path, "w") as f:
        for name, data in all_res.items():
            f.write(f"{'='*55}\n")
            f.write(f"  {name}\n")
            f.write(f"{'='*55}\n\n")
            for split in ["val", "test"]:
                f.write(f"--- {split.upper()} ---\n")
                f.write(data["results"][split]["report"])
                f.write("\n")
    print(f"[SAVE] Classification reports saved to {path.name}")


# ─────────────────────────────────────────────
# 7. Feature Importance (Random Forest)
# ─────────────────────────────────────────────
def plot_feature_importance(all_res):
    rf    = all_res["Random Forest"]["model"]
    imp   = rf.feature_importances_
    top_n = 20

    top_idx = np.argsort(imp)[::-1][:top_n]
    top_imp = imp[top_idx]

    # Build feature names
    feat_names = []
    for prefix in ["MFCC", "Delta", "Delta2"]:
        for stat in ["mean", "std"]:
            for i in range(40):
                feat_names.append(f"{prefix}_{i+1}_{stat}")
    for name in ["centroid_mean", "centroid_std",
                 "rolloff_mean",  "rolloff_std",
                 "bandwidth_mean","bandwidth_std",
                 "flatness_mean",
                 "zcr_mean", "zcr_std",
                 "rms_mean", "rms_std"]:
        feat_names.append(name)

    top_names = [feat_names[i] if i < len(feat_names) else f"feat_{i}"
                 for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), top_imp[::-1],
                   color="#4A90D9", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top_n} most important features (Random Forest)",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = OUT_RES / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved feature_importance.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Omli — Day 2: Train & Benchmark Classifiers")
    print("=" * 55)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    X_train, X_val, X_test, scaler = normalize(X_train, X_val, X_test)

    all_res = train_all(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n[OUTPUT] Saving results...")
    plot_confusion_matrices(all_res)
    save_benchmark(all_res)
    save_classification_reports(all_res)
    plot_feature_importance(all_res)

    print("\n✓ Day 2 complete. Ready for Day 3 — child vs adult deep dive.")


if __name__ == "__main__":
    main()