"""
Omli — Doro Reliability Layer
Module 3: Combined API
========================
Wires Module 1 (Hallucination Detector) and
Module 2 (Dropout Predictor) into a single
drop-in layer for Doro's pipeline.

Every time a child speaks, call:
  result = analyze_turn(audio, conversation_history)

Returns a single JSON with everything Doro needs
to decide what to do next.
"""

import json
import time
import warnings
import numpy as np
from pathlib import Path

from day6_hallucination_detector import (
    load_hallucination_detector,
    predict_trust,
)
from day6_dropout_predictor import (
    load_dropout_predictor,
    predict_dropout,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "outputs" / "day6"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Load Models (call once at startup)
# ─────────────────────────────────────────────
print("[INIT] Loading Doro Reliability Layer...")
_hall_model,  _hall_scaler,  _whisper = load_hallucination_detector()
_drop_model,  _drop_scaler            = load_dropout_predictor()
print("[INIT] All models loaded. Ready.")


# ─────────────────────────────────────────────
# Core API
# ─────────────────────────────────────────────
def analyze_turn(audio, conversation_history,
                 trust_threshold=0.50,
                 dropout_threshold=0.55):
    """
    Main entry point for the Doro Reliability Layer.

    Call this after every child utterance.

    Args:
        audio               : np.ndarray — raw audio at 16kHz
        conversation_history: list of dicts, each with:
                              {"audio": np.ndarray,
                               "response_gap_ms": float}
        trust_threshold     : below this → ask child to repeat
        dropout_threshold   : above this → intervene

    Returns:
        dict — full reliability report for this turn
    """
    t_start = time.perf_counter()

    # ── Module 1: Hallucination Detection ──
    hall_result = predict_trust(
        audio, _hall_model, _hall_scaler, _whisper,
        threshold=trust_threshold
    )

    # ── Module 2: Dropout Prediction ──
    drop_result = predict_dropout(
        conversation_history, _drop_model, _drop_scaler,
        threshold=dropout_threshold
    )

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # ── Decide recommended action ──
    # Priority: hallucination > dropout > normal
    if hall_result["trust_flag"] == "repeat_request":
        final_action = "ask_child_to_repeat"
        alert_level  = "medium"
    elif drop_result["dropout_flag"] == "intervene":
        final_action = drop_result["recommended_action"]
        alert_level  = "high" \
                       if drop_result["dropout_probability"] >= 0.75 \
                       else "medium"
    else:
        final_action = "continue_lesson"
        alert_level  = "none"

    return {
        "turn":              len(conversation_history) + 1,
        "latency_ms":        latency_ms,
        "alert_level":       alert_level,
        "recommended_action": final_action,

        "transcription": {
            "transcript":     hall_result["transcript"],
            "trust_score":    hall_result["trust_score"],
            "trust_flag":     hall_result["trust_flag"],
            "avg_logprob":    hall_result["avg_logprob"],
            "no_speech_prob": hall_result["no_speech_prob"],
            "snr_db":         hall_result["snr_estimate_db"],
        },

        "engagement": {
            "dropout_probability": drop_result["dropout_probability"],
            "dropout_flag":        drop_result["dropout_flag"],
            "energy_trend":        drop_result.get("energy_trend", "unknown"),
            "gap_trend":           drop_result.get("gap_trend",    "unknown"),
            "avg_gap_ms":          drop_result.get("avg_response_gap_ms", 0),
        },

        "doro_instruction": _build_doro_instruction(
            hall_result, drop_result, final_action
        ),
    }


def _build_doro_instruction(hall_result, drop_result, action):
    """
    Human-readable instruction for Doro's response engine.
    This is what the AI response layer would actually consume.
    """
    instructions = {
        "ask_child_to_repeat": (
            "Child's speech was unclear. Ask them to repeat naturally: "
            "'Sorry, I didn't quite catch that — can you say it again?'"
        ),
        "immediate_intervention": (
            "Child is about to disengage. Switch strategy immediately: "
            "tell a short joke, ask their favourite topic, or "
            "say 'want to try something fun?'"
        ),
        "change_topic_or_simplify": (
            "Child engagement dropping. Simplify the current question "
            "or smoothly transition to a new topic."
        ),
        "add_encouragement": (
            "Child showing early disengagement signs. "
            "Add positive reinforcement: 'You're doing really well!'"
        ),
        "continue_lesson": (
            "All signals normal. Continue current lesson flow."
        ),
    }
    return instructions.get(action, "Continue normally.")


# ─────────────────────────────────────────────
# Batch Analysis (for demo / testing)
# ─────────────────────────────────────────────
def analyze_session(audio_clips_with_gaps):
    """
    Run the reliability layer over a full session.

    Args:
        audio_clips_with_gaps: list of {"audio": np.ndarray,
                                        "response_gap_ms": float}
    Returns:
        list of per-turn results
    """
    history = []
    results = []

    for clip in audio_clips_with_gaps:
        result = analyze_turn(
            clip["audio"],
            history,
        )
        results.append(result)
        history.append({
            "audio":           clip["audio"],
            "response_gap_ms": clip["response_gap_ms"],
        })

    return results


def print_session_report(results):
    """Pretty print a full session report."""
    print("\n" + "=" * 65)
    print("  DORO RELIABILITY LAYER — SESSION REPORT")
    print("=" * 65)
    print(f"  {'Turn':<6} {'Trust':>7} {'Dropout%':>10} "
          f"{'Alert':>8}  Action")
    print("  " + "-" * 60)

    for r in results:
        turn    = r["turn"]
        trust   = r["transcription"]["trust_score"]
        dropout = r["engagement"]["dropout_probability"]
        alert   = r["alert_level"]
        action  = r["recommended_action"][:30]
        latency = r["latency_ms"]

        alert_icon = {"none":"✓","medium":"⚠","high":"🔴"}.get(alert,"?")
        print(f"  {turn:<6} {trust:>7.2f} {dropout:>10.2f} "
              f"  {alert_icon} {alert:<6}  {action}")

    print("  " + "-" * 60)

    # Summary
    trust_scores   = [r["transcription"]["trust_score"] for r in results]
    dropout_probs  = [r["engagement"]["dropout_probability"] for r in results]
    latencies      = [r["latency_ms"] for r in results]
    interventions  = sum(1 for r in results
                         if r["alert_level"] in ["medium","high"])

    print(f"\n  Turns analyzed    : {len(results)}")
    print(f"  Avg trust score   : {np.mean(trust_scores):.2f}")
    print(f"  Peak dropout prob : {max(dropout_probs):.2f}")
    print(f"  Interventions     : {interventions}")
    print(f"  Avg latency       : {np.mean(latencies):.1f} ms")
    print(f"  Total latency     : {sum(latencies):.1f} ms")
    print("=" * 65)