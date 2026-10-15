"""
Driver Drowsiness Detection - Streamlit dashboard.

Run it with:
    streamlit run app.py

How the live video works
------------------------
The webcam is captured by the *browser* and streamed to Python over WebRTC
(`streamlit-webrtc`). Every frame goes through `DrowsinessPipeline.process_frame`
on a worker thread and the annotated frame is streamed straight back, so the
driver sees themselves - with the YOLO boxes, labels and heatmap drawn on top -
at full frame rate. Because the camera belongs to the browser, the same app
works on localhost and when it is deployed to a server with no webcam.

The numbers around the video (status, gauge, tiles, charts) live in
`st.fragment(run_every=...)` blocks that read the latest snapshot from the
worker thread a few times per second. Only those fragments re-run, so the video
stream is never interrupted.

There is no upload control anywhere in this file by design: the system only
ever looks at the live camera.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from aiortc.codecs import h264 as _h264, vpx as _vpx
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# aiortc encodes the video it sends back to the browser at 0.5-1.5 Mbps by
# default, which makes an HD webcam look blurry. Allow a much higher bitrate.
_vpx.DEFAULT_BITRATE, _vpx.MAX_BITRATE = 4_000_000, 10_000_000
_h264.DEFAULT_BITRATE, _h264.MAX_BITRATE = 4_000_000, 10_000_000

VIDEO_QUALITY = {
    "HD 720p (recommended)": (1280, 720),
    "Full HD 1080p": (1920, 1080),
    "SD 480p (fastest)": (640, 480),
}

# Make `src` importable no matter which directory streamlit was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path              # noqa: E402
from src.detector import ModelError                           # noqa: E402
from src.drowsiness import DrowsinessMetrics, DrowsinessState  # noqa: E402
from src.overlay import draw_error_frame                      # noqa: E402
from src.pipeline import DrowsinessPipeline                   # noqa: E402

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Status colours are reserved for the driver state and always shown with a label.
STATE_STYLE = {
    DrowsinessState.ALERT: ("#22c55e", "ALERT", "Driver is awake and attentive", "🟢"),
    DrowsinessState.WARNING: ("#f59e0b", "WARNING", "Early signs of fatigue", "🟠"),
    DrowsinessState.DROWSY: ("#ef4444", "DROWSY", "Wake up! Pull over safely", "🔴"),
    DrowsinessState.NO_FACE: ("#64748b", "NO FACE", "Look at the camera", "⚪"),
}
ACCENT = "#38bdf8"
INK_MUTED = "#94a3b8"
GRID = "rgba(148,163,184,0.15)"

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(148,163,184,0.12); }

      .hero { display:flex; align-items:center; justify-content:space-between;
              gap:1rem; flex-wrap:wrap; margin-bottom: 1.1rem; }
      .hero h1 { font-size: 1.9rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
      .hero p  { margin: 0.2rem 0 0; color: #94a3b8; font-size: 0.95rem; }
      .chip { display:inline-block; padding: 0.28rem 0.7rem; border-radius: 999px;
              font-size: 0.75rem; font-weight: 600; margin-left: 0.35rem;
              background: rgba(56,189,248,0.10); color: #7dd3fc;
              border: 1px solid rgba(56,189,248,0.25); }

      .card { background: #111827; border: 1px solid rgba(148,163,184,0.14);
              border-radius: 16px; padding: 1rem 1.1rem; }
      .card-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
                    color: #94a3b8; font-weight: 600; margin-bottom: 0.6rem; }

      .status { border-radius: 16px; padding: 1rem 1.2rem; color: #fff;
                display:flex; align-items:center; gap: 0.9rem; }
      .status .icon { font-size: 2rem; line-height: 1; }
      .status .label { font-size: 1.6rem; font-weight: 800; letter-spacing: 0.05em; line-height: 1.1; }
      .status .sub { font-size: 0.9rem; opacity: 0.9; }
      .status.pulse { animation: pulse 0.8s infinite; }
      @keyframes pulse { 0%,100% { filter: brightness(1); } 50% { filter: brightness(1.35); } }

      .tile { background: #111827; border: 1px solid rgba(148,163,184,0.14);
              border-radius: 14px; padding: 0.7rem 0.9rem; height: 100%;
              border-top: 3px solid var(--edge, #334155); }
      .tile .k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em;
                 color: #94a3b8; font-weight: 600; }
      .tile .v { font-size: 1.55rem; font-weight: 700; color: #f1f5f9; line-height: 1.25;
                 font-variant-numeric: tabular-nums; }
      .tile .h { font-size: 0.72rem; color: #64748b; }

      .idle { text-align:center; padding: 2.2rem 1rem; color: #94a3b8; }
      .idle .big { font-size: 2.6rem; }
      .steps { text-align:left; display:inline-block; margin-top: 0.6rem; line-height: 1.9; }

      /* The camera panel */
      iframe[title="streamlit_webrtc.component.webrtc_streamer"] { border-radius: 16px; }

      /* Browser-side alarm player: plays but stays out of sight. */
      [data-testid="stAudio"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Live session: shared between the Streamlit script and the WebRTC worker thread
# -----------------------------------------------------------------------------
@dataclass
class LiveSnapshot:
    """The latest results, copied out of the worker thread for the UI."""

    metrics: DrowsinessMetrics
    fps: float
    inference_ms: float
    total_ms: float
    heatmap_view: Optional[np.ndarray]
    alarm_count: int
    frames: int
    updated_at: float = field(default_factory=time.time)


class LiveSession:
    """Owns one pipeline per browser session and guards it with a lock."""

    def __init__(self, pipeline: DrowsinessPipeline, flip: bool):
        self.pipeline = pipeline
        self.flip = flip
        self.lock = threading.Lock()
        self.snapshot: Optional[LiveSnapshot] = None
        self.error: Optional[str] = None

    def process(self, image: np.ndarray) -> np.ndarray:
        if self.flip:
            image = cv2.flip(image, 1)   # mirror view, so it feels natural to the driver
        try:
            with self.lock:
                result = self.pipeline.process_frame(image)
                self.snapshot = LiveSnapshot(
                    metrics=result.metrics,
                    fps=result.fps,
                    inference_ms=result.inference_ms,
                    total_ms=result.total_ms,
                    heatmap_view=result.heatmap_view,
                    alarm_count=self.pipeline.alarm.trigger_count,
                    frames=self.pipeline.frames_processed,
                )
                self.error = None
            return result.frame
        except Exception as exc:  # noqa: BLE001 - never kill the stream on one bad frame
            self.error = f"Processing error: {exc}"
            height, width = image.shape[:2]
            return draw_error_frame(width, height, self.error)

    def read(self) -> Tuple[Optional[LiveSnapshot], List[Tuple[float, float]], Dict]:
        """Snapshot + recent score history + session summary, read atomically."""
        with self.lock:
            history = list(self.pipeline.engine.history[-900:])
            summary = self.pipeline.engine.session_summary()
            return self.snapshot, history, summary

    def reset(self) -> None:
        with self.lock:
            self.pipeline.reset_session()
            self.snapshot = None

    def apply_settings(self, settings: Dict) -> None:
        with self.lock:
            p = self.pipeline
            p.engine.ear_threshold = settings["ear_threshold"]
            p.engine.auto_calibrate = settings["auto_calibrate"]
            p.engine.mar_threshold = settings["mar_threshold"]
            p.engine.yawn_alarm_duration = settings["yawn_alarm_duration"]
            p.engine.microsleep_duration = settings["microsleep_duration"]
            p.engine.head_drop_angle = settings["head_drop_angle"]
            p.engine.head_drop_duration = settings["head_drop_duration"]
            p.detector.confidence = settings["yolo_confidence"]
            p.heatmap.alpha = settings["heatmap_alpha"]
            p.cfg.heatmap.enabled = settings["show_heatmap"]
            p.cfg.ui.show_boxes = settings["show_boxes"]
            p.cfg.ui.show_landmarks = settings["show_landmarks"]
            p.cfg.ui.show_hud = settings["show_hud"]
            # The server speaker only plays when chosen; the browser alarm is
            # driven from the trigger counter instead.
            p.alarm.enabled = settings["alarm_enabled"] and settings["alarm_on_server"]


class DrowsinessProcessor(VideoProcessorBase):
    """streamlit-webrtc hands every incoming frame to `recv`."""

    def __init__(self, live: LiveSession):
        self.live = live

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated = self.live.process(image)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def tile(label: str, value: str, hint: str = "", edge: str = "#334155") -> str:
    return (
        f'<div class="tile" style="--edge:{edge}"><div class="k">{label}</div>'
        f'<div class="v">{value}</div><div class="h">{hint or "&nbsp;"}</div></div>'
    )


def render_tiles(items: List[str]) -> None:
    for column, html in zip(st.columns(len(items)), items):
        column.markdown(html, unsafe_allow_html=True)


def status_banner(state: DrowsinessState, score: Optional[float] = None) -> str:
    color, label, sub, icon = STATE_STYLE[state]
    pulse = " pulse" if state == DrowsinessState.DROWSY else ""
    score_html = (
        f'<div style="margin-left:auto;text-align:right"><div class="sub">Score</div>'
        f'<div class="label">{score:.0f}</div></div>' if score is not None else ""
    )
    return (
        f'<div class="status{pulse}" style="background:{color}">'
        f'<div class="icon">{icon}</div><div><div class="label">{label}</div>'
        f'<div class="sub">{sub}</div></div>{score_html}</div>'
    )


def plot_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=12),
        showlegend=False,
        hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9", bordercolor="#334155"),
    )
    return fig


def gauge_figure(score: float, state: DrowsinessState, warn: float, drowsy: float) -> go.Figure:
    color = STATE_STYLE[state][0]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix=" / 100", font=dict(size=30, color="#f1f5f9")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=INK_MUTED, tickwidth=1,
                          tickvals=[0, warn, drowsy, 100]),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, warn], color="rgba(34,197,94,0.16)"),
                    dict(range=[warn, drowsy], color="rgba(245,158,11,0.18)"),
                    dict(range=[drowsy, 100], color="rgba(239,68,68,0.20)"),
                ],
            ),
        )
    )
    return plot_layout(fig, 210)


def timeline_figure(history: List[Tuple[float, float]], warn: float, drowsy: float) -> go.Figure:
    xs = [t for t, _ in history]
    ys = [s for _, s in history]
    fig = go.Figure(
        go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color=ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.10)",
            hovertemplate="%{x:.1f}s · score %{y:.0f}<extra></extra>",
        )
    )
    for level, label, color in ((warn, "Warning", "#f59e0b"), (drowsy, "Drowsy", "#ef4444")):
        fig.add_hline(
            y=level, line=dict(color=color, width=1, dash="dot"),
            annotation_text=f"{label} {level:.0f}", annotation_position="top left",
            annotation_font=dict(color="#cbd5e1", size=11),
        )
    fig.update_xaxes(title="Session time (s)", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(range=[0, 100], gridcolor=GRID, zeroline=False, title="Score")
    fig.update_layout(hovermode="x unified")
    return plot_layout(fig, 280)


def contributions_figure(contributions: Dict[str, float], weights: Dict[str, float]) -> go.Figure:
    names = {"perclos": "PERCLOS", "closure": "Current eye closure",
             "yawn": "Yawning", "blink": "Blink rate"}
    keys = [k for k in ("perclos", "closure", "yawn", "blink")]
    values = [contributions.get(k, 0.0) * weights.get(k, 0.0) * 100 for k in keys]
    fig = go.Figure(
        go.Bar(
            y=[names[k] for k in keys], x=values, orientation="h",
            marker=dict(color=ACCENT, cornerradius=4),
            text=[f"{v:.1f}" for v in values], textposition="outside",
            textfont=dict(color="#cbd5e1"),
            hovertemplate="%{y}: %{x:.1f} points<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 45], gridcolor=GRID, title="Points added to the score")
    fig.update_yaxes(autorange="reversed")
    return plot_layout(fig, 240)


@st.cache_data(show_spinner=False)
def alarm_wav_bytes(path: str) -> Optional[bytes]:
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def build_live_session(config) -> Tuple[Optional[LiveSession], Optional[str]]:
    try:
        pipeline = DrowsinessPipeline(config)
        return LiveSession(pipeline, flip=bool(config.camera.flip_horizontal)), None
    except ModelError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - MediaPipe missing, bad config, ...
        return None, f"Could not initialise the detection pipeline:\n\n{exc}"


# -----------------------------------------------------------------------------
# Configuration + per-browser-session pipeline
# -----------------------------------------------------------------------------
try:
    cfg = load_config()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"**Configuration problem**\n\n{exc}")
    st.stop()

st.session_state.setdefault("alarm_seen", 0)
st.session_state.setdefault("alarm_until", 0.0)
st.session_state.setdefault("last_summary", None)

if "live" not in st.session_state:
    with st.spinner("Loading the YOLO model and MediaPipe FaceMesh…"):
        live, init_error = build_live_session(cfg)
    st.session_state.live = live
    st.session_state.init_error = init_error

live: Optional[LiveSession] = st.session_state.live
if live is None:
    st.error(f"**The detector could not start**\n\n{st.session_state.init_error}")
    st.markdown(
        "**Common fixes**\n"
        "- Missing model: run `python tools/download_models.py`\n"
        "- Missing packages: run `pip install -r requirements.txt` (Python 3.10–3.12)"
    )
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚗 Control panel")
    st.caption("Settings apply instantly, even while the camera is running.")

    with st.expander("🎯 Detection", expanded=True):
        auto_calibrate = st.toggle(
            "Auto-calibrate to my eyes", value=bool(cfg.drowsiness.get("auto_calibrate", True)),
            help="Learns your normal open-eye EAR in the first seconds and sets the "
                 "'closed' threshold from it. Recommended.",
        )
        ear_threshold = st.slider(
            "Eye closed below (EAR)", 0.10, 0.40, float(cfg.drowsiness.ear_threshold), 0.01,
            disabled=auto_calibrate,
            help="Used when auto-calibration is off (and for the first second before it kicks in).",
        )
        mar_threshold = st.slider(
            "Yawn above (MAR)", 0.20, 1.20, float(cfg.drowsiness.mar_threshold), 0.05,
            help="Open your mouth wide and read MAR on the Mouth card; set this a bit below it.",
        )
        yawn_alarm_duration = st.slider(
            "Alarm when mouth open for (s)", 1.5, 8.0,
            float(cfg.drowsiness.get("yawn_alarm_duration", 2.5)), 0.5,
            help="A yawn shows WARNING; holding the mouth open this long raises the alarm.",
        )
        microsleep_duration = st.slider(
            "Micro-sleep after (s)", 0.5, 4.0, float(cfg.drowsiness.microsleep_duration), 0.1,
            help="Eyes closed longer than this forces DROWSY immediately.",
        )
        head_drop_angle = st.slider(
            "Head falling at (degrees)", 10, 45,
            int(cfg.drowsiness.get("head_drop_angle", 20)), 1,
            help="How far the head must drop forward from your normal posture.",
        )
        head_drop_duration = st.slider(
            "Alarm when head down for (s)", 0.5, 4.0,
            float(cfg.drowsiness.get("head_drop_duration", 1.5)), 0.1,
            help="Head dropped forward, back or sideways this long raises the alarm.",
        )
        yolo_confidence = st.slider(
            "YOLO confidence", 0.10, 0.90, float(cfg.model.confidence), 0.05,
        )

    with st.expander("📷 Camera", expanded=True):
        quality = st.selectbox(
            "Video quality", list(VIDEO_QUALITY), index=0,
            help="Takes effect the next time you press Start.",
        )

    with st.expander("🎨 Video overlay", expanded=True):
        show_boxes = st.toggle("YOLO boxes & labels", value=bool(cfg.ui.show_boxes))
        show_heatmap = st.toggle("Attention heatmap", value=bool(cfg.heatmap.enabled))
        heatmap_alpha = st.slider("Heatmap strength", 0.0, 1.0, float(cfg.heatmap.alpha), 0.05,
                                  disabled=not show_heatmap)
        show_landmarks = st.toggle("Face mesh points", value=bool(cfg.ui.show_landmarks))
        show_hud = st.toggle("On-video HUD panel", value=bool(cfg.ui.show_hud))

    with st.expander("🔊 Alarm", expanded=True):
        alarm_enabled = st.toggle("Alarm enabled", value=bool(cfg.alarm.enabled))
        # On localhost the computer's own speakers are the most reliable (no
        # browser autoplay rules). When deployed, only the browser can reach the user.
        host = st.context.headers.get("Host", "") if st.context.headers else ""
        running_locally = host.startswith(("localhost", "127.0.0.1", "[::1]"))
        alarm_target = st.radio(
            "Play the alarm on",
            ["This browser", "The computer running the app"],
            index=1 if running_locally else 0,
            disabled=not alarm_enabled,
            help="Use 'This browser' when the app is deployed on a server.",
        )
        alarm_bytes = alarm_wav_bytes(str(resolve_path(cfg.alarm.sound_file)))
        if st.button("🔔 Test alarm", width="stretch", disabled=not alarm_enabled):
            if alarm_target == "This browser":
                st.session_state.alarm_until = time.time() + 2.0   # status panel plays it
            else:
                live.pipeline.alarm.test()
            st.toast("Alarm played")

    if st.button("🔁 Reset session counters", width="stretch"):
        live.reset()
        st.session_state.alarm_seen = 0
        st.session_state.alarm_until = 0.0
        st.toast("Counters reset")

    st.divider()
    with st.expander("ℹ️ System information"):
        st.json(live.pipeline.describe())

live.apply_settings(
    dict(
        ear_threshold=ear_threshold,
        auto_calibrate=auto_calibrate,
        mar_threshold=mar_threshold,
        yawn_alarm_duration=yawn_alarm_duration,
        microsleep_duration=microsleep_duration,
        head_drop_angle=float(head_drop_angle),
        head_drop_duration=head_drop_duration,
        yolo_confidence=yolo_confidence,
        heatmap_alpha=heatmap_alpha,
        show_heatmap=show_heatmap,
        show_boxes=show_boxes,
        show_landmarks=show_landmarks,
        show_hud=show_hud,
        alarm_enabled=alarm_enabled,
        alarm_on_server=alarm_target != "This browser",
    )
)
browser_alarm = alarm_enabled and alarm_target == "This browser"

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div>
        <h1>🚗 Driver Drowsiness Detection</h1>
        <p>Real-time driver monitoring from your webcam: face detection, eye and mouth
        tracking, a fatigue score and an alarm.</p>
      </div>
      <div>
        <span class="chip">YOLO11 face</span>
        <span class="chip">MediaPipe FaceMesh</span>
        <span class="chip">EAR · MAR · PERCLOS</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

camera_col, status_col = st.columns([3, 2], gap="large")

# -----------------------------------------------------------------------------
# Live camera (browser webcam -> Python -> annotated video back to the page)
# -----------------------------------------------------------------------------
with camera_col:
    st.markdown('<div class="card-title">Live camera · driver view</div>', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="driver-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: DrowsinessProcessor(live),
        media_stream_constraints={
            "video": {
                "width": {"ideal": VIDEO_QUALITY[quality][0]},
                "height": {"ideal": VIDEO_QUALITY[quality][1]},
                "frameRate": {"ideal": int(cfg.camera.fps)},
            },
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_html_attrs={
            "style": {"width": "100%", "borderRadius": "14px", "background": "#020617"},
            "autoPlay": True,
            "controls": False,
            "muted": True,
        },
        translations={"start": "▶  Start monitoring", "stop": "⏹  Stop"},
        async_processing=True,
    )
    if not ctx.state.playing:
        st.markdown(
            """
            <div class="card idle">
              <div class="big">📷</div>
              <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0">Camera is off</div>
              <div class="steps">
                1. Press <b>Start monitoring</b> above<br>
                2. Allow camera access when the browser asks<br>
                3. Sit facing the camera in good light
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Live panels: refreshed on a timer without touching the video stream
# -----------------------------------------------------------------------------
def current_data():
    if not ctx.state.playing:
        return None, [], None
    return live.read()


@st.fragment(run_every=0.4)
def status_panel() -> None:
    snapshot, _, summary = current_data()

    # Browser alarm: start playing on every new trigger and keep the same
    # element on the page long enough for the sound to finish.
    if browser_alarm:
        now = time.time()
        if snapshot is not None and snapshot.alarm_count > st.session_state.alarm_seen:
            st.session_state.alarm_seen = snapshot.alarm_count
            st.session_state.alarm_until = now + 2.0
        if now < st.session_state.alarm_until and alarm_bytes:
            st.audio(alarm_bytes, format="audio/wav", autoplay=True)

    if snapshot is None:
        state_label = "CONNECTING…" if ctx.state.playing else "STANDBY"
        st.markdown(
            f'<div class="status" style="background:#1e293b"><div class="icon">⏸️</div>'
            f'<div><div class="label">{state_label}</div>'
            f'<div class="sub">Start the camera to begin monitoring</div></div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            gauge_figure(0, DrowsinessState.NO_FACE, live.pipeline.engine.warning_score,
                         live.pipeline.engine.drowsy_score),
            width="stretch", config={"displayModeBar": False},
        )
        return

    m = snapshot.metrics
    st.session_state.last_summary = summary
    st.markdown(status_banner(m.state, m.score), unsafe_allow_html=True)

    if live.error:
        st.error(live.error)
    elif m.state in (DrowsinessState.DROWSY, DrowsinessState.WARNING):
        engine = live.pipeline.engine
        if m.head_down:
            reason = (f"😴 Head falling {m.head_direction} for **{m.head_down_duration:.1f}s** "
                      f"(alarm at {engine.head_drop_duration:.1f}s)")
        elif m.closure_duration > engine.blink_max_duration:
            reason = f"👁️ Eyes closed for **{m.closure_duration:.1f}s**"
        elif m.is_yawning:
            reason = (f"🥱 Mouth open for **{m.yawn_duration:.1f}s** "
                      f"(alarm at {engine.yawn_alarm_duration:.1f}s)")
        elif engine.recent_yawns >= engine.frequent_yawns:
            reason = f"🥱 Frequent yawning — **{engine.recent_yawns}** in the last minute"
        else:
            reason = f"👁️ Eyes closed **{m.perclos * 100:.0f}%** of the last minute"
        if m.state == DrowsinessState.DROWSY:
            st.error(f"🔊 **ALARM** — {reason} · triggered {snapshot.alarm_count}× this session")
        else:
            st.warning(f"⚠️ **Fatigue sign** — {reason}")

    st.plotly_chart(
        gauge_figure(m.score, m.state, live.pipeline.engine.warning_score,
                     live.pipeline.engine.drowsy_score),
        width="stretch", config={"displayModeBar": False},
    )

    closed_edge = "#ef4444" if m.eyes_closed else "#22c55e"
    render_tiles([
        tile("Eyes", "Closed" if m.eyes_closed else "Open",
             f"{m.closure_duration:.2f}s shut" if m.closure_duration > 0
             else f"EAR {m.ear:.2f} · closed &lt; {m.ear_threshold:.2f}"
             + (" ✓" if m.calibrated else ""), closed_edge),
        tile("Mouth", "Yawning" if m.is_yawning else "Normal",
             f"MAR {m.mar:.2f} · yawn &gt; {live.pipeline.engine.mar_threshold:.2f}",
             "#f59e0b" if m.is_yawning else "#334155"),
    ])
    st.write("")
    render_tiles([
        tile("PERCLOS", f"{m.perclos * 100:.0f}%", "eyes-closed time, last 60s"),
        tile("Head", f"Down ({m.head_direction})" if m.head_down else "Upright",
             f"pitch {m.head_pitch:+.0f}° · roll {m.head_tilt:+.0f}°"
             + (" ✓" if m.head_calibrated else " · learning posture"),
             "#ef4444" if m.head_down else "#334155"),
    ])



with status_col:
    st.markdown('<div class="card-title">Driver status</div>', unsafe_allow_html=True)
    status_panel()

st.write("")


@st.fragment(run_every=1.0)
def details_panel() -> None:
    snapshot, history, summary = current_data()
    m = snapshot.metrics if snapshot else None

    st.markdown('<div class="card-title">Session statistics</div>', unsafe_allow_html=True)
    if m is not None:
        render_tiles([
            tile("Blinks", f"{m.blink_count}", f"{m.blink_rate:.0f} per minute"),
            tile("Yawns", f"{m.yawn_count}", f"{m.yawn_rate:.1f} per minute",
                 "#f59e0b" if m.yawn_count else "#334155"),
            tile("Micro-sleeps", f"{m.microsleep_count}", f"longest {m.longest_closure:.1f}s",
                 "#ef4444" if m.microsleep_count else "#334155"),
            tile("Head drops", f"{m.head_nod_count}", "nodding off",
                 "#ef4444" if m.head_nod_count else "#334155"),
            tile("Alarms", f"{snapshot.alarm_count}", "this session",
                 "#ef4444" if snapshot.alarm_count else "#334155"),
            tile("Speed", f"{snapshot.fps:.0f} FPS", f"YOLO {snapshot.inference_ms:.0f} ms",
                 "#22c55e" if snapshot.fps >= 15 else "#f59e0b"),
        ])
    else:
        last = st.session_state.get("last_summary")
        if last and last.get("duration_seconds", 0) > 1:
            st.caption("Last session")
            render_tiles([
                tile("Duration", f"{last['duration_seconds']:.0f}s"),
                tile("Blinks", f"{last['total_blinks']}"),
                tile("Yawns", f"{last['total_yawns']}"),
                tile("Micro-sleeps", f"{last['microsleeps']}"),
                tile("Head drops", f"{last.get('head_nods', 0)}"),
                tile("Longest closure", f"{last['longest_closure']:.1f}s"),
                tile("Peak score", f"{last['peak_score']:.0f}"),
            ])
        else:
            st.info("Statistics appear here once monitoring starts.")

    st.write("")
    timeline_tab, heatmap_tab, breakdown_tab, about_tab = st.tabs(
        ["📈 Score timeline", "🔥 Attention heatmap", "🧮 Score breakdown", "📘 How it works"]
    )
    engine = live.pipeline.engine
    with timeline_tab:
        if len(history) > 4:
            st.plotly_chart(
                timeline_figure(history, engine.warning_score, engine.drowsy_score),
                width="stretch", config={"displayModeBar": False},
            )
        else:
            st.caption("The drowsiness score over time will be plotted here.")
    with heatmap_tab:
        if snapshot is not None and snapshot.heatmap_view is not None:
            left, right = st.columns([2, 3])
            left.image(snapshot.heatmap_view, channels="BGR", width="stretch")
            right.markdown(
                "Warm colours show where fatigue signals are coming from:\n\n"
                "- **Eyes** heat up as the eye aspect ratio drops toward closed\n"
                "- **Mouth** heats up while yawning\n"
                "- **Whole face** glows as the overall score rises"
            )
        else:
            st.caption("Turn on *Attention heatmap* in the sidebar and start the camera.")
    with breakdown_tab:
        if m is not None:
            st.plotly_chart(
                contributions_figure(m.contributions, engine.weights),
                width="stretch", config={"displayModeBar": False},
            )
            st.caption("How many points each signal adds to the 0–100 drowsiness score right now.")
        else:
            st.caption("Shows what is driving the score once monitoring starts.")
    with about_tab:
        st.markdown(
            f"""
| Step | What happens |
|---|---|
| 1. Face | **YOLO11** finds the driver's face in each frame |
| 2. Landmarks | **MediaPipe FaceMesh** places 478 points on that face |
| 3. Eyes | **EAR** (eye aspect ratio) below **{engine.active_ear_threshold:.2f}** = eye closed{" (calibrated to your eyes)" if engine.calibrated else ""}. Blinks under {engine.blink_max_duration:.1f}s never raise the score |
| 4. Mouth | **MAR** (mouth aspect ratio) above **{engine.mar_threshold:.2f}** for {engine.yawn_min_duration:.1f}s = yawn → **WARNING**. Held open {engine.yawn_alarm_duration:.1f}s → **DROWSY + alarm**. 3 yawns in a minute → WARNING |
| 5. Head | Head drops **{engine.head_drop_angle:.0f}°** forward (or {engine.head_back_angle:.0f}° back / {engine.head_side_angle:.0f}° sideways) from your normal posture → **WARNING**. Down for {engine.head_drop_duration:.1f}s → **DROWSY + alarm**, even if the face falls out of view |
| 6. Score | PERCLOS, current closure, yawns and blink rate become a 0–100 score |
| 7. Alarm | Score ≥ **{engine.drowsy_score:.0f}**, eyes shut > **{engine.microsleep_duration:.1f}s**, mouth open > **{engine.yawn_alarm_duration:.1f}s**, or head down > **{engine.head_drop_duration:.1f}s** sounds the alarm |
"""
        )


details_panel()
