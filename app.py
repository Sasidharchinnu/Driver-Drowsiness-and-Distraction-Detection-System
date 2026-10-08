"""
Driver Drowsiness Detection - Streamlit dashboard.

Run it with:
    streamlit run app.py

The webcam starts automatically. There is no upload control anywhere in this
file by design: the system only ever looks at the live camera.

How the live video works in Streamlit
-------------------------------------
Streamlit normally runs a script top to bottom and then stops. To get live
video we keep a `while` loop at the bottom of the script and write each new
frame into a placeholder created with `st.empty()`. Pressing any button makes
Streamlit re-run the script, which breaks out of the loop naturally - that is
how Stop, Reset and the sliders take effect.

The pipeline object lives in `st.session_state` so it survives those re-runs;
otherwise we would reload the YOLO model several times a second.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Make `src` importable no matter which directory streamlit was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import CameraError, list_available_cameras   # noqa: E402
from src.config import load_config                           # noqa: E402
from src.detector import ModelError                          # noqa: E402
from src.drowsiness import DrowsinessState                   # noqa: E402
from src.overlay import draw_error_frame                     # noqa: E402
from src.pipeline import DrowsinessPipeline                  # noqa: E402

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A little CSS to make the status banner big and the metric cards tidy.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 1rem; }
      .status-banner {
          padding: 0.9rem 1.2rem; border-radius: 10px; text-align: center;
          font-size: 1.9rem; font-weight: 800; letter-spacing: 0.06em;
          color: #fff; margin-bottom: 0.6rem;
      }
      .status-alert   { background: #15803d; }
      .status-warning { background: #b45309; }
      .status-drowsy  { background: #b91c1c; animation: flash 0.7s infinite; }
      .status-noface  { background: #44403c; }
      @keyframes flash { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }
      .metric-card {
          background: rgba(120,120,120,0.10); border-radius: 8px;
          padding: 0.55rem 0.75rem; border-left: 3px solid #666;
      }
      .metric-label { font-size: 0.72rem; text-transform: uppercase;
                      letter-spacing: 0.05em; opacity: 0.70; }
      .metric-value { font-size: 1.45rem; font-weight: 700; line-height: 1.25; }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_CSS_CLASS = {
    DrowsinessState.ALERT: "status-alert",
    DrowsinessState.WARNING: "status-warning",
    DrowsinessState.DROWSY: "status-drowsy",
    DrowsinessState.NO_FACE: "status-noface",
}

STATUS_MESSAGE = {
    DrowsinessState.ALERT: "ALERT — driver is awake",
    DrowsinessState.WARNING: "WARNING — early fatigue signs",
    DrowsinessState.DROWSY: "DROWSY — WAKE UP!",
    DrowsinessState.NO_FACE: "NO FACE — please face the camera",
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def metric_card(label: str, value: str, color: str = "#666") -> str:
    """Render one stat tile as HTML."""
    return (
        f'<div class="metric-card" style="border-left-color:{color}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>'
    )


def state_hex(state: DrowsinessState) -> str:
    return {
        DrowsinessState.ALERT: "#22c55e",
        DrowsinessState.WARNING: "#f59e0b",
        DrowsinessState.DROWSY: "#ef4444",
        DrowsinessState.NO_FACE: "#78716c",
    }.get(state, "#78716c")


@st.cache_data(show_spinner=False)
def cached_camera_list() -> list:
    """Probing cameras takes a moment, so only do it once per session."""
    return list_available_cameras(max_index=4)


def build_pipeline(config, camera_index: int):
    """Create the pipeline and store it in session state.

    Returns (pipeline, error_message). On failure the message is shown to the
    user with instructions rather than a raw traceback.
    """
    try:
        pipeline = DrowsinessPipeline(config, camera_index=camera_index)
        return pipeline, None
    except ModelError as exc:
        return None, str(exc)
    except Exception as exc:  # MediaPipe missing, corrupted config, etc.
        return None, f"Could not initialise the detection pipeline:\n\n{exc}"


def shutdown_pipeline() -> None:
    """Release the camera and MediaPipe cleanly."""
    pipeline = st.session_state.get("pipeline")
    if pipeline is not None:
        pipeline.close()
    st.session_state.pipeline = None
    st.session_state.running = False


# -----------------------------------------------------------------------------
# Load the configuration (a bad config is fatal, so report it and stop)
# -----------------------------------------------------------------------------
try:
    cfg = load_config()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"**Configuration problem**\n\n{exc}")
    st.stop()

# Session state defaults. Streamlit re-runs this script constantly, so every
# value we want to survive a re-run has to live here.
st.session_state.setdefault("pipeline", None)
st.session_state.setdefault("running", False)
st.session_state.setdefault("error", None)
st.session_state.setdefault("camera_index", cfg.camera.index)
st.session_state.setdefault("autostart_done", False)
st.session_state.setdefault("last_summary", None)

# -----------------------------------------------------------------------------
# Sidebar: controls and diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🚗 Controls")

    cameras = cached_camera_list()
    if cameras:
        default_index = cameras.index(st.session_state.camera_index) if (
            st.session_state.camera_index in cameras
        ) else 0
        chosen_camera = st.selectbox(
            "Camera device",
            options=cameras,
            index=default_index,
            format_func=lambda i: f"Camera {i}" + (" (built-in)" if i == 0 else ""),
            disabled=st.session_state.running,
            help="Detected automatically. Stop the camera to change it.",
        )
        st.session_state.camera_index = chosen_camera
    else:
        st.warning("No camera detected. Connect one, then press **Re-scan**.")
        if st.button("🔄 Re-scan for cameras", width="stretch"):
            cached_camera_list.clear()
            st.rerun()

    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("▶ Start", width="stretch", type="primary",
                     disabled=st.session_state.running or not cameras):
            st.session_state.running = True
            st.session_state.error = None
            st.rerun()
    with stop_col:
        if st.button("⏹ Stop", width="stretch", disabled=not st.session_state.running):
            pipeline = st.session_state.get("pipeline")
            if pipeline is not None:
                st.session_state.last_summary = pipeline.engine.session_summary()
            shutdown_pipeline()
            st.rerun()

    if st.button("🔁 Reset counters", width="stretch"):
        pipeline = st.session_state.get("pipeline")
        if pipeline is not None:
            pipeline.reset_session()
        st.toast("Counters reset")

    st.divider()

    # ---- live-tunable detection settings ----
    st.subheader("Detection settings")
    ear_threshold = st.slider(
        "EAR threshold (eye closed below this)",
        0.10, 0.40, float(cfg.drowsiness.ear_threshold), 0.01,
        help="Lower it if normal blinking is flagged; raise it if closures are missed.",
    )
    mar_threshold = st.slider(
        "MAR threshold (yawn above this)",
        0.30, 1.20, float(cfg.drowsiness.mar_threshold), 0.05,
    )
    microsleep_duration = st.slider(
        "Micro-sleep duration (s)",
        0.5, 4.0, float(cfg.drowsiness.microsleep_duration), 0.1,
        help="Eyes closed longer than this forces DROWSY immediately.",
    )
    yolo_confidence = st.slider(
        "YOLO confidence threshold",
        0.10, 0.90, float(cfg.model.confidence), 0.05,
    )

    st.divider()
    st.subheader("Display")
    show_heatmap = st.checkbox("Attention heatmap overlay", value=bool(cfg.heatmap.enabled))
    heatmap_alpha = st.slider("Heatmap strength", 0.0, 1.0, float(cfg.heatmap.alpha), 0.05)
    show_boxes = st.checkbox("YOLO boxes & labels", value=bool(cfg.ui.show_boxes))
    show_landmarks = st.checkbox("Face mesh points", value=bool(cfg.ui.show_landmarks))
    show_hud = st.checkbox("On-frame HUD", value=bool(cfg.ui.show_hud))

    st.divider()
    st.subheader("Alarm")
    alarm_enabled = st.checkbox("Alarm enabled", value=bool(cfg.alarm.enabled))
    if st.button("🔊 Test alarm", width="stretch"):
        pipeline = st.session_state.get("pipeline")
        if pipeline is not None:
            pipeline.alarm.test()
            st.toast("Alarm played")
        else:
            # Let the user test the sound before ever starting the camera.
            from src.alarm import AlarmSystem

            AlarmSystem(cfg.alarm).test()
            st.toast("Alarm played")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("Driver Drowsiness Detection")
st.caption(
    "Live webcam · YOLO face detection · MediaPipe EAR/MAR · real-time drowsiness scoring"
)

status_placeholder = st.empty()
alert_placeholder = st.empty()

video_column, side_column = st.columns([3, 2], gap="medium")
with video_column:
    st.markdown("##### Live camera — YOLO detections")
    video_placeholder = st.empty()
with side_column:
    st.markdown("##### Attention heatmap")
    heatmap_placeholder = st.empty()

metrics_placeholder = st.empty()
chart_placeholder = st.empty()
info_placeholder = st.empty()

# -----------------------------------------------------------------------------
# Start the camera automatically the first time the page loads.
# This is the "no manual input" requirement: the user does not have to click
# anything for detection to begin.
# -----------------------------------------------------------------------------
if not st.session_state.autostart_done and cameras:
    st.session_state.autostart_done = True
    st.session_state.running = True

# -----------------------------------------------------------------------------
# Build the pipeline if we are supposed to be running but do not have one yet.
# -----------------------------------------------------------------------------
if st.session_state.running and st.session_state.pipeline is None:
    with st.spinner("Loading the YOLO model and opening the webcam…"):
        pipeline, error = build_pipeline(cfg, st.session_state.camera_index)
        if error:
            st.session_state.error = error
            st.session_state.running = False
        else:
            try:
                pipeline.start_camera()
                st.session_state.pipeline = pipeline
            except CameraError as exc:
                pipeline.close()
                st.session_state.error = str(exc)
                st.session_state.running = False

# -----------------------------------------------------------------------------
# Error state
# -----------------------------------------------------------------------------
if st.session_state.error:
    status_placeholder.markdown(
        '<div class="status-banner status-noface">SYSTEM ERROR</div>', unsafe_allow_html=True
    )
    alert_placeholder.error(st.session_state.error)
    video_placeholder.image(
        draw_error_frame(cfg.camera.width, cfg.camera.height, st.session_state.error),
        channels="BGR",
        width="stretch",
    )
    with info_placeholder.container():
        st.markdown(
            "**Common fixes**\n"
            "- **Camera permission (macOS):** System Settings → Privacy & Security → "
            "Camera → enable Terminal/iTerm/your IDE, then *fully quit and reopen it*.\n"
            "- **Camera busy:** close Zoom, Teams, FaceTime or Photo Booth.\n"
            "- **Missing model:** run `python tools/download_models.py`.\n"
            "- **Wrong device:** pick a different camera in the sidebar."
        )
    st.stop()

# -----------------------------------------------------------------------------
# Idle state (user pressed Stop)
# -----------------------------------------------------------------------------
if not st.session_state.running:
    status_placeholder.markdown(
        '<div class="status-banner status-noface">CAMERA STOPPED</div>', unsafe_allow_html=True
    )
    video_placeholder.info("Press **▶ Start** in the sidebar to begin monitoring.")

    summary = st.session_state.get("last_summary")
    if summary:
        with metrics_placeholder.container():
            st.markdown("##### Last session summary")
            columns = st.columns(6)
            items = [
                ("Duration", f"{summary['duration_seconds']:.0f} s"),
                ("Blinks", f"{summary['total_blinks']}"),
                ("Yawns", f"{summary['total_yawns']}"),
                ("Micro-sleeps", f"{summary['microsleeps']}"),
                ("Longest closure", f"{summary['longest_closure']:.2f} s"),
                ("Peak score", f"{summary['peak_score']:.0f}"),
            ]
            for column, (label, value) in zip(columns, items):
                column.markdown(metric_card(label, value), unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------------------
# Running: apply the sidebar settings to the live objects
# -----------------------------------------------------------------------------
pipeline = st.session_state.pipeline
pipeline.engine.ear_threshold = ear_threshold
pipeline.engine.mar_threshold = mar_threshold
pipeline.engine.microsleep_duration = microsleep_duration
pipeline.detector.confidence = yolo_confidence
pipeline.heatmap.enabled = show_heatmap
pipeline.heatmap.alpha = heatmap_alpha
pipeline.alarm.enabled = alarm_enabled
pipeline.cfg.heatmap.enabled = show_heatmap   # the pipeline re-reads this every frame
pipeline.cfg.ui.show_boxes = show_boxes
pipeline.cfg.ui.show_landmarks = show_landmarks
pipeline.cfg.ui.show_hud = show_hud

with info_placeholder.expander("System information", expanded=False):
    st.json(pipeline.describe())
    if not pipeline.detector.has_eye_classes:
        st.caption(
            "This checkpoint detects the class **face** only, so eye and mouth regions come "
            "from MediaPipe FaceMesh landmarks inside the YOLO face box. Train the 5-class "
            "model (`python training/train.py`) to have YOLO detect eyes and mouth natively."
        )

# -----------------------------------------------------------------------------
# THE LIVE LOOP
# -----------------------------------------------------------------------------
# Throttling: the video is updated every frame, but redrawing every metric
# widget 30 times a second would swamp Streamlit's websocket. Numbers refresh
# ~8x/second and the chart ~1x/second, which still looks instant to a human.
METRICS_EVERY = 4
CHART_EVERY = 30

# Smoke-test hook: setting DROWSINESS_MAX_FRAMES makes the loop stop after that
# many frames, so `python tests/smoke_test.py` can run the whole dashboard
# head-less without hanging forever. It is unset during normal use.
max_frames = int(os.environ.get("DROWSINESS_MAX_FRAMES", "0"))

frame_counter = 0
target_frame_time = 1.0 / max(float(cfg.ui.target_fps), 1.0)

try:
    while st.session_state.running:
        loop_started = time.perf_counter()

        result = pipeline.read_and_process()
        if result is None:
            time.sleep(0.005)   # no new frame yet; yield the CPU briefly
            continue

        frame_counter += 1
        metrics = result.metrics
        color = state_hex(metrics.state)

        # ---- video + heatmap ----
        video_placeholder.image(result.frame, channels="BGR", width="stretch")
        if show_heatmap and result.heatmap_view is not None:
            heatmap_placeholder.image(
                result.heatmap_view, channels="BGR", width="stretch"
            )
        elif frame_counter == 1:
            heatmap_placeholder.info("Heatmap overlay is switched off in the sidebar.")

        # ---- status banner + alarm indicator ----
        if frame_counter % METRICS_EVERY == 0 or frame_counter == 1:
            status_placeholder.markdown(
                f'<div class="status-banner {STATUS_CSS_CLASS[metrics.state]}">'
                f"{STATUS_MESSAGE[metrics.state]} &nbsp;·&nbsp; {metrics.score:.0f}/100</div>",
                unsafe_allow_html=True,
            )

            if metrics.state == DrowsinessState.DROWSY:
                alert_placeholder.error(
                    f"🔊 **ALARM ACTIVE** — eyes closed {metrics.closure_duration:.1f}s · "
                    f"score {metrics.score:.0f} · triggered {pipeline.alarm.trigger_count}x "
                    f"this session"
                )
            elif metrics.state == DrowsinessState.WARNING:
                alert_placeholder.warning(
                    f"⚠️ Fatigue building — PERCLOS {metrics.perclos * 100:.0f}% · "
                    f"{metrics.yawn_count} yawns"
                )
            else:
                alert_placeholder.empty()

            # ---- metric cards ----
            with metrics_placeholder.container():
                row1 = st.columns(5)
                row1[0].markdown(
                    metric_card("Drowsiness score", f"{metrics.score:.1f}", color),
                    unsafe_allow_html=True,
                )
                row1[1].markdown(
                    metric_card(
                        "Eye closure",
                        f"{metrics.closure_duration:.2f} s",
                        "#ef4444" if metrics.eyes_closed else "#22c55e",
                    ),
                    unsafe_allow_html=True,
                )
                row1[2].markdown(
                    metric_card("Blinks", f"{metrics.blink_count}"), unsafe_allow_html=True
                )
                row1[3].markdown(
                    metric_card(
                        "Yawns",
                        f"{metrics.yawn_count}",
                        "#ef4444" if metrics.is_yawning else "#666",
                    ),
                    unsafe_allow_html=True,
                )
                row1[4].markdown(
                    metric_card(
                        "FPS",
                        f"{result.fps:.1f}",
                        "#22c55e" if result.fps >= 15 else "#f59e0b",
                    ),
                    unsafe_allow_html=True,
                )

                row2 = st.columns(5)
                row2[0].markdown(
                    metric_card("EAR", f"{metrics.ear:.3f}"), unsafe_allow_html=True
                )
                row2[1].markdown(
                    metric_card("MAR", f"{metrics.mar:.3f}"), unsafe_allow_html=True
                )
                row2[2].markdown(
                    metric_card("PERCLOS", f"{metrics.perclos * 100:.1f}%"),
                    unsafe_allow_html=True,
                )
                row2[3].markdown(
                    metric_card(
                        "Micro-sleeps",
                        f"{metrics.microsleep_count}",
                        "#ef4444" if metrics.microsleep_count else "#666",
                    ),
                    unsafe_allow_html=True,
                )
                row2[4].markdown(
                    metric_card("Faces detected", f"{metrics.face_count}"),
                    unsafe_allow_html=True,
                )

        # ---- score history chart ----
        if frame_counter % CHART_EVERY == 0 and len(pipeline.engine.history) > 4:
            recent = pipeline.engine.history[-600:]      # roughly the last 20 s
            chart_placeholder.line_chart(
                {"Drowsiness score": [score for _, score in recent]},
                height=180,
                color="#ef4444",
            )

        # ---- pace the loop so we do not burn 100% CPU ----
        elapsed = time.perf_counter() - loop_started
        if elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)

        if max_frames and frame_counter >= max_frames:
            break   # smoke-test mode only

except CameraError as exc:
    # The camera vanished mid-session (unplugged, or grabbed by another app).
    shutdown_pipeline()
    st.session_state.error = str(exc)
    st.rerun()
except Exception as exc:  # noqa: BLE001 - last line of defence for the UI
    shutdown_pipeline()
    st.session_state.error = f"Unexpected error in the processing loop:\n\n{exc}"
    st.rerun()
