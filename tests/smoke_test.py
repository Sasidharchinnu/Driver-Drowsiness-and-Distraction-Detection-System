"""
End-to-end smoke test: proves the whole system runs on this machine.

It checks, in order:
  1. config.yaml loads and validates
  2. the YOLO checkpoint is present and runs inference
  3. MediaPipe FaceMesh initialises
  4. the alarm sound exists / can be generated
  5. the webcam opens and delivers frames
  6. the full pipeline processes real frames at a usable frame rate
  7. the Streamlit dashboard script runs top to bottom without errors

Run it with:
    python tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS, FAIL = "  PASS", "  FAIL"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


def main() -> int:
    print("\nDriver Drowsiness Detection - smoke test\n" + "=" * 52)

    # 1. configuration -------------------------------------------------------
    print("\n[1] Configuration")
    from src.config import load_config

    cfg = load_config()
    check("config.yaml loads and validates", True, f"EAR threshold {cfg.drowsiness.ear_threshold}")

    # 2. YOLO ----------------------------------------------------------------
    print("\n[2] YOLO model")
    from src.detector import YOLODetector

    try:
        detector = YOLODetector(
            weights=cfg.model.weights,
            device=cfg.model.device,
            confidence=cfg.model.confidence,
            iou=cfg.model.iou,
            imgsz=cfg.model.imgsz,
        )
        info = detector.describe()
        check("YOLO checkpoint loads", True, f"{info['weights']} on {info['device']}")
        check("model exposes classes", bool(info["classes"]), str(info["classes"]))
    except Exception as exc:
        check("YOLO checkpoint loads", False, str(exc)[:120])
        print("\n  -> Run: python tools/download_models.py")
        return 1

    # 3. landmarks -----------------------------------------------------------
    print("\n[3] MediaPipe FaceMesh")
    try:
        from src.landmarks import FaceMeshAnalyzer

        mesh = FaceMeshAnalyzer()
        mesh.close()
        check("FaceMesh initialises", True)
    except Exception as exc:
        check("FaceMesh initialises", False, str(exc)[:120])

    # 4. alarm ---------------------------------------------------------------
    print("\n[4] Alarm")
    from src.alarm import AlarmSystem

    alarm = AlarmSystem(cfg.alarm)
    described = alarm.describe()
    check("alarm sound available", Path(described["sound_file"]).exists(), described["backend"])

    # 5. camera --------------------------------------------------------------
    print("\n[5] Webcam")
    from src.camera import CameraError, list_available_cameras

    cameras = list_available_cameras(max_index=3)
    if not check("a camera was found", bool(cameras), f"indices {cameras}"):
        print("\n  -> Check camera permission: System Settings > Privacy & Security > Camera")
        return 1

    # 6. pipeline ------------------------------------------------------------
    print("\n[6] Live pipeline")
    from src.pipeline import DrowsinessPipeline

    pipeline = DrowsinessPipeline(cfg, camera_index=cameras[0])
    try:
        pipeline.start_camera()
        check("camera opens", True, str(pipeline.describe().get("camera")))

        processed, faces_seen, started = 0, 0, time.time()
        result = None
        while processed < 60 and time.time() - started < 20:
            result = pipeline.read_and_process()
            if result is None:
                time.sleep(0.005)
                continue
            processed += 1
            if result.detections.face_count:
                faces_seen += 1

        check("frames processed", processed >= 30, f"{processed} frames")
        check("frame rate is usable", result is not None and result.fps >= 10,
              f"{result.fps:.1f} FPS" if result else "no frames")
        check("annotated frame produced", result is not None and result.frame is not None)
        check("heatmap produced", result is not None and result.heatmap_view is not None)
        print(f"        (a face was detected in {faces_seen}/{processed} frames - "
              f"0 is fine if nobody is in front of the camera)")
    finally:
        pipeline.close()
        check("camera released cleanly", True)

    # 7. the Streamlit app script -------------------------------------------
    print("\n[7] Streamlit dashboard script")
    try:
        from streamlit.testing.v1 import AppTest

        os.environ["DROWSINESS_MAX_FRAMES"] = "12"   # stop the live loop quickly
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
        app.run()
        check("app.py runs without exceptions", not app.exception,
              str(app.exception[0].value)[:120] if app.exception else "")
    except ImportError:
        check("streamlit testing harness available", False, "streamlit too old")
    finally:
        os.environ.pop("DROWSINESS_MAX_FRAMES", None)

    # summary ----------------------------------------------------------------
    print("\n" + "=" * 52)
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("All checks passed. Start the app with:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
