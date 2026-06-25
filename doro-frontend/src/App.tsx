// @ts-ignore
import "./App.css";
import { useState, useRef, useCallback } from "react";
import axios from "axios";
import {
  Upload,
  Mic,
  MicOff,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  MessageSquare,
  Brain,
} from "lucide-react";

// ─── Types ───────────────────────────────────────
interface TranscriptionResult {
  transcript: string;
  trust_score: number;
  trust_flag: string;
  avg_logprob: number;
  no_speech_prob: number;
  snr_db: number;
}

interface EngagementResult {
  dropout_probability: number;
  dropout_flag: string;
  energy_trend: string;
  gap_trend: string;
  avg_gap_ms: number;
}

interface AnalysisResult {
  turn: number;
  latency_ms: number;
  alert_level: string;
  recommended_action: string;
  transcription: TranscriptionResult;
  engagement: EngagementResult;
  doro_instruction: string;
}

// ─── Helpers ─────────────────────────────────────
const API = "http://localhost:8000";

function ScoreBar({
  score,
  color,
  label,
}: {
  score: number;
  color: string;
  label: string;
}) {
  const pct = Math.round(score * 100);
  return (
    <div className="score-bar-container">
      <div className="score-bar-header">
        <span className="score-label">{label}</span>
        <span className="score-value" style={{ color }}>
          {score.toFixed(2)}
        </span>
      </div>
      <div className="score-track">
        <div
          className="score-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function AlertBadge({ level }: { level: string }) {
  const config = {
    none: {
      icon: <CheckCircle size={14} />,
      label: "All clear",
      cls: "badge-green",
    },
    medium: {
      icon: <AlertTriangle size={14} />,
      label: "Warning",
      cls: "badge-yellow",
    },
    high: {
      icon: <XCircle size={14} />,
      label: "Intervene",
      cls: "badge-red",
    },
  };
  const c = config[level as keyof typeof config] || config.none;
  return (
    <span className={`badge ${c.cls}`}>
      {c.icon}
      {c.label}
    </span>
  );
}

// ─── Main App ─────────────────────────────────────
export default function App() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // ── File Upload ──
  const handleFile = useCallback(
    async (file: File) => {
      setFileName(file.name);
      setError(null);
      setLoading(true);
      setResult(null);

      const form = new FormData();
      form.append("file", file);

      try {
        const res = await axios.post<AnalysisResult>(`${API}/analyze`, form, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setResult(res.data);
      } catch (e: any) {
        setError(
          e?.response?.data?.detail ||
            "Could not reach the backend. Is the API running?"
        );
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  // ── Mic Recording ──
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", {
          type: "audio/webm",
        });
        handleFile(file);
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
    } catch {
      setError("Microphone access denied.");
    }
  };

  const stopRecording = () => {
    mediaRef.current?.stop();
    setRecording(false);
  };

  // ── Render ──
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <Brain size={28} color="#4A90D9" />
            <div>
              <h1>Doro Reliability Layer</h1>
              <p>Real-time child speech analysis for Omli</p>
            </div>
          </div>
          <div className="header-tags">
            <span className="tag">Hallucination Detector</span>
            <span className="tag">Dropout Predictor</span>
          </div>
        </div>
      </header>

      <main className="main">
        {/* Upload Section */}
        <section className="card upload-card">
          <div className="section-title">
            <Upload size={18} />
            <span>Audio Input</span>
          </div>

          <div
            className="drop-zone"
            onDrop={onDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileRef.current?.click()}
          >
            <Upload size={32} color="#4A90D9" />
            <p className="drop-title">
              Drop a WAV file here or click to browse
            </p>
            <p className="drop-sub">Supports WAV, MP3, FLAC, WEBM</p>
            {fileName && (
              <p className="file-name">📎 {fileName}</p>
            )}
            <input
              ref={fileRef}
              type="file"
              accept="audio/*"
              style={{ display: "none" }}
              onChange={onFileChange}
            />
          </div>

          <div className="divider">
            <span>or</span>
          </div>

          <button
            className={`mic-btn ${recording ? "mic-active" : ""}`}
            onClick={recording ? stopRecording : startRecording}
          >
            {recording ? (
              <>
                <MicOff size={18} />
                Stop Recording
              </>
            ) : (
              <>
                <Mic size={18} />
                Record from Mic
              </>
            )}
          </button>
          {recording && (
            <p className="recording-hint">
              🔴 Recording... click Stop when done
            </p>
          )}
        </section>

        {/* Loading */}
        {loading && (
          <div className="card loading-card">
            <div className="spinner" />
            <div>
              <p className="loading-title">Analyzing audio...</p>
              <p className="loading-sub">
                Running Whisper + hallucination detector + dropout predictor
              </p>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="card error-card">
            <XCircle size={20} color="#E74C3C" />
            <p>{error}</p>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="results">
            {/* Status Bar */}
            <div className="status-bar">
              <div className="status-left">
                <AlertBadge level={result.alert_level} />
                <span className="latency">
                  ⚡ {result.latency_ms.toFixed(0)}ms
                </span>
              </div>
              <span className="action-tag">
                → {result.recommended_action.replace(/_/g, " ")}
              </span>
            </div>

            <div className="results-grid">
              {/* Transcription Card */}
              <div className="card result-card">
                <div className="section-title">
                  <MessageSquare size={18} color="#4A90D9" />
                  <span>Transcription</span>
                </div>

                <div className="transcript-box">
                  <p className="transcript-text">
                    "{result.transcription.transcript || "— no speech detected —"}"
                  </p>
                </div>

                <ScoreBar
                  score={result.transcription.trust_score}
                  color={
                    result.transcription.trust_score > 0.7
                      ? "#27AE60"
                      : result.transcription.trust_score > 0.5
                      ? "#F39C12"
                      : "#E74C3C"
                  }
                  label="Trust Score"
                />

                <div className="meta-grid">
                  <div className="meta-item">
                    <span className="meta-label">SNR</span>
                    <span className="meta-value">
                      {result.transcription.snr_db.toFixed(1)} dB
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Whisper logprob</span>
                    <span className="meta-value">
                      {result.transcription.avg_logprob.toFixed(3)}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">No-speech prob</span>
                    <span className="meta-value">
                      {result.transcription.no_speech_prob.toFixed(3)}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Flag</span>
                    <span
                      className={`flag ${
                        result.transcription.trust_flag === "trust"
                          ? "flag-green"
                          : "flag-red"
                      }`}
                    >
                      {result.transcription.trust_flag}
                    </span>
                  </div>
                </div>
              </div>

              {/* Engagement Card */}
              <div className="card result-card">
                <div className="section-title">
                  <Activity size={18} color="#E67E22" />
                  <span>Engagement</span>
                </div>

                <ScoreBar
                  score={result.engagement.dropout_probability}
                  color={
                    result.engagement.dropout_probability < 0.35
                      ? "#27AE60"
                      : result.engagement.dropout_probability < 0.6
                      ? "#F39C12"
                      : "#E74C3C"
                  }
                  label="Dropout Risk"
                />

                <div className="meta-grid">
                  <div className="meta-item">
                    <span className="meta-label">Energy trend</span>
                    <span
                      className={`flag ${
                        result.engagement.energy_trend === "rising"
                          ? "flag-green"
                          : result.engagement.energy_trend === "stable"
                          ? "flag-yellow"
                          : "flag-red"
                      }`}
                    >
                      {result.engagement.energy_trend}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Gap trend</span>
                    <span
                      className={`flag ${
                        result.engagement.gap_trend === "stable"
                          ? "flag-green"
                          : "flag-red"
                      }`}
                    >
                      {result.engagement.gap_trend}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Avg gap</span>
                    <span className="meta-value">
                      {result.engagement.avg_gap_ms} ms
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Flag</span>
                    <span
                      className={`flag ${
                        result.engagement.dropout_flag === "continue"
                          ? "flag-green"
                          : "flag-red"
                      }`}
                    >
                      {result.engagement.dropout_flag}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Doro Instruction */}
            <div className="card instruction-card">
              <div className="section-title">
                <Brain size={18} color="#9B59B6" />
                <span>Doro Instruction</span>
              </div>
              <p className="instruction-text">
                {result.doro_instruction}
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}