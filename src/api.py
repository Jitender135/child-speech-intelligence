"""
Omli — Doro Reliability Layer API
===================================
FastAPI server that wraps the reliability layer
so the React frontend can call it.

Run:
  uvicorn src.api:app --reload --port 8000
"""

import sys
import numpy as np
import librosa
import tempfile
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from day6_doro_reliability_layer import analyze_turn

app = FastAPI(title="Doro Reliability Layer API")

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "Doro Reliability Layer is running"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Accept a WAV/MP3 audio file, run it through the
    Doro Reliability Layer, return full JSON result.
    """
    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load audio
        audio, _ = librosa.load(tmp_path, sr=16000, mono=True, duration=5.0)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        audio = audio.astype(np.float32)

        # Run reliability layer
        result = analyze_turn(audio, conversation_history=[])
        return result

    finally:
        os.unlink(tmp_path)


@app.post("/analyze-session")
async def analyze_session_endpoint(files: list[UploadFile] = File(...)):
    """
    Accept multiple audio files as a session.
    Returns per-turn analysis for the full session.
    """
    from day6_doro_reliability_layer import analyze_session

    clips = []
    tmp_paths = []

    for f in files:
        suffix = Path(f.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await f.read()
            tmp.write(content)
            tmp_paths.append(tmp.name)

        audio, _ = librosa.load(tmp_paths[-1], sr=16000,
                                  mono=True, duration=5.0)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        clips.append({
            "audio": audio.astype(np.float32),
            "response_gap_ms": 500.0,
        })

    try:
        results = analyze_session(clips)
        return {"turns": results, "total_turns": len(results)}
    finally:
        for p in tmp_paths:
            os.unlink(p)