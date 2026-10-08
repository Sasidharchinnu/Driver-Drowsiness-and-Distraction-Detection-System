"""
Command-line version of the detector: a plain OpenCV window, no browser.

Why this exists alongside app.py
--------------------------------
  * It is the fastest way to demo the system (no web server, lowest latency).
  * It is the best way to prove the dashboard is not hiding anything - the same
    pipeline, the same numbers, just a different display.
  * If Streamlit ever misbehaves on a machine, this still works.

Run:
    python run_cli.py

Keys:
    q / ESC   quit
    r         reset the counters
    h         toggle the heatmap
    l         toggle the face-mesh points
    b         toggle the YOLO boxes
    space     test the alarm
    s         save a screenshot to logs/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import CameraError, list_available_cameras   # noqa: E402
from src.config import load_config                           # noqa: E402
from src.detector import ModelError                          # noqa: E402
from src.pipeline import DrowsinessPipeline                  # noqa: E402

WINDOW_NAME = "Driver Drowsiness Detection  [q]uit  [r]eset  [h]eatmap  [b]oxes  [l]andmarks"


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time drowsiness detection (OpenCV window)")
    parser.add_argument("--camera", type=int, default=None, help="camera index")
    parser.add_argument("--weights", type=str, default=None, help="override the YOLO weights")
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--show-heatmap-window", action="store_true",
                        help="open a second window with the standalone heatmap")
    args = parser.parse_args()

    # ---- configuration ----
    try:
        cfg = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 1

    if args.weights:
        cfg.model.weights = args.weights
    if args.no_heatmap:
        cfg.heatmap.enabled = False

    # ---- pipeline ----
    try:
        pipeline = DrowsinessPipeline(cfg, camera_index=args.camera)
    except ModelError as exc:
        print(f"Model error:\n{exc}", file=sys.stderr)
        return 1

    print("Loading…")
    print(f"  model   : {pipeline.detector.describe()}")

    # ---- camera ----
    try:
        pipeline.start_camera()
    except CameraError as exc:
        print(f"\nCamera error:\n{exc}", file=sys.stderr)
        available = list_available_cameras(4)
        if available:
            print(f"\nTry one of these indices: python run_cli.py --camera {available[0]}")
        pipeline.close()
        return 1

    print(f"  camera  : {pipeline.describe().get('camera')}")
    print("\nRunning. Press 'q' in the window to quit.\n")

    screenshot_dir = PROJECT_ROOT / "logs"
    screenshot_dir.mkdir(exist_ok=True)
    last_state = None

    try:
        while True:
            try:
                result = pipeline.read_and_process()
            except CameraError as exc:
                print(f"\nCamera lost: {exc}", file=sys.stderr)
                break

            if result is None:
                time.sleep(0.005)
                continue

            cv2.imshow(WINDOW_NAME, result.frame)
            if args.show_heatmap_window and result.heatmap_view is not None:
                cv2.imshow("Attention heatmap", result.heatmap_view)

            # Print a line only when the state CHANGES, so the terminal stays
            # readable instead of scrolling 30 lines a second.
            if result.metrics.state != last_state:
                metrics = result.metrics
                print(
                    f"[{time.strftime('%H:%M:%S')}] {metrics.state.value:8s} "
                    f"score={metrics.score:5.1f}  EAR={metrics.ear:.3f}  MAR={metrics.mar:.3f}  "
                    f"blinks={metrics.blink_count}  yawns={metrics.yawn_count}  "
                    f"fps={result.fps:.1f}"
                )
                last_state = metrics.state

            # ---- keyboard ----
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # q or ESC
                break
            if key == ord("r"):
                pipeline.reset_session()
                print("  -> counters reset")
            elif key == ord("h"):
                cfg.heatmap.enabled = not cfg.heatmap.enabled
                print(f"  -> heatmap {'on' if cfg.heatmap.enabled else 'off'}")
            elif key == ord("l"):
                cfg.ui.show_landmarks = not cfg.ui.show_landmarks
                print(f"  -> landmarks {'on' if cfg.ui.show_landmarks else 'off'}")
            elif key == ord("b"):
                cfg.ui.show_boxes = not cfg.ui.show_boxes
                print(f"  -> boxes {'on' if cfg.ui.show_boxes else 'off'}")
            elif key == ord(" "):
                pipeline.alarm.test()
                print("  -> alarm test")
            elif key == ord("s"):
                path = screenshot_dir / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), result.frame)
                print(f"  -> saved {path}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Always release the camera and close the windows, whatever happened.
        pipeline.close()
        cv2.destroyAllWindows()

    summary = pipeline.engine.session_summary()
    print("\n" + "=" * 46)
    print("SESSION SUMMARY")
    print(f"  Duration        {summary['duration_seconds']:.0f} s")
    print(f"  Blinks          {summary['total_blinks']}")
    print(f"  Yawns           {summary['total_yawns']}")
    print(f"  Micro-sleeps    {summary['microsleeps']}")
    print(f"  Longest closure {summary['longest_closure']:.2f} s")
    print(f"  Avg blink rate  {summary['avg_blink_rate']:.1f} /min")
    print(f"  Peak score      {summary['peak_score']:.1f}")
    print(f"  Alarms fired    {pipeline.alarm.trigger_count}")
    print("=" * 46)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
