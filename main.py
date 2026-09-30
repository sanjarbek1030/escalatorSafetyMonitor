"""
main.py - Escalator Safety Monitor (automated visual warning system).

Run:  python main.py
      python main.py --input input/other.mp4 --output output/other_result.mp4 --no-preview
"""
from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path

import cv2

import config as cfg
from alert_manager import AlertManager
from behavior_analyzer import BehaviorAnalyzer
from crowd_analyzer import CrowdAnalyzer
from detector import PersonDetector, select_device
from escalator import EscalatorRegion
from event_logger import EventLogger
from person_state import DANGER_STATES, BehaviorState
from pose_estimator import PoseEstimator
from tracker import TrackManager
from utils import format_timestamp
from visualization import Visualizer

GREEN, ORANGE, RED = (80, 220, 80), (0, 190, 255), (60, 60, 255)


class EscalatorSafetyMonitor:
    def __init__(self, input_path: Path, output_path: Path, show_preview: bool):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.show_preview = show_preview
        self.last_frame_index = -1
        self.last_timestamp = 0.0
        self._error_count = 0
        self._crowd_events = {}

    # ------------------------------------------------------------------ video
    @staticmethod
    def _read_video_properties(capture):
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:                     # metadata missing: measure a real frame
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Video has no readable frames.")
            height, width = frame.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not fps or not math.isfinite(fps) or fps <= 0 or fps > 240:
            print(f"[WARN] Invalid FPS metadata ({fps}); using fallback {cfg.FALLBACK_FPS}")
            fps = cfg.FALLBACK_FPS
        return width, height, float(fps), total

    # ------------------------------------------------------------------ run
    def run(self) -> int:
        print("=" * 64)
        print(" ESCALATOR SAFETY MONITOR - automated visual warning system")
        print(" (heuristic analysis for HUMAN REVIEW; false alarms and misses are possible)")
        print("=" * 64)

        capture = cv2.VideoCapture(str(self.input_path))
        if not capture.isOpened():
            print(f"[ERROR] Cannot open video: {self.input_path}")
            print("        Put your video at input/escalator_input.mp4 or use --input <path>.")
            return 1
        try:
            width, height, fps, total = self._read_video_properties(capture)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            capture.release()
            return 1
        print(f"[VIDEO] {self.input_path.name}: {width}x{height} @ {fps:.2f} FPS"
              + (f", {total} frames (~{total / fps:.1f} s)" if total > 0 else ""))

        device, is_cuda = select_device(cfg.FORCE_CPU)
        use_half = cfg.USE_HALF_PRECISION and is_cuda
        try:
            print(f"[MODEL] Loading detector {cfg.MODEL_PATH} and pose model {cfg.POSE_MODEL_PATH} ...")
            self.detector = PersonDetector(cfg.MODEL_PATH, device, use_half)
            self.pose_estimator = PoseEstimator(cfg.POSE_MODEL_PATH, device, use_half)
        except Exception as exc:                         # noqa: BLE001 - friendly message for beginners
            print(f"[ERROR] Could not load a YOLO model: {exc}")
            print("        Check your internet connection (weights are downloaded on first use) "
                  "and that 'pip install ultralytics' succeeded.")
            capture.release()
            return 1

        try:
            self.escalator = EscalatorRegion(cfg.ESCALATOR_POLYGON, cfg.ESCALATOR_DIRECTION,
                                             width, height, cfg.REFERENCE_FRAME_SIZE)
        except ValueError as exc:
            print(f"[ERROR] Invalid escalator configuration: {exc}")
            capture.release()
            return 1
        self.pose_crop = self.escalator.bounding_rect(cfg.POSE_CROP_MARGIN_FRACTION)
        self.track_manager = TrackManager(self.escalator, fps)
        self.crowd = CrowdAnalyzer(fps)
        self.analyzer = BehaviorAnalyzer()
        self.alert_manager = AlertManager()
        self.event_logger = EventLogger(cfg.EVENTS_JSON, cfg.EVENT_SCREENSHOT_DIR, self.input_path.name, fps)
        self.visualizer = Visualizer(self.escalator, width, height)
        self.fps = fps

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"[ERROR] Cannot create output folder: {exc}")
            capture.release()
            return 1
        fourcc = cv2.VideoWriter_fourcc(*cfg.OUTPUT_FOURCC)
        writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[ERROR] VideoWriter could not open {self.output_path} with codec '{cfg.OUTPUT_FOURCC}'.")
            print("        Try OUTPUT_FOURCC = 'avc1' or 'XVID' (and .avi) in config.py.")
            capture.release()
            return 1

        print("[RUN] Processing... (press 'q' in the preview window to stop)")
        frame_index, smoothed_fps, started = 0, 0.0, time.perf_counter()
        preview_enabled = self.show_preview
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break                                     # end of video
                loop_start = time.perf_counter()
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                annotated = self._process_frame(frame, frame_index, smoothed_fps)
                writer.write(annotated)

                if preview_enabled:
                    try:
                        cv2.imshow("Escalator Safety Monitor", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("[RUN] Stopped by user.")
                            break
                    except cv2.error:
                        print("[WARN] Preview window unavailable; continuing without it.")
                        preview_enabled = False

                elapsed = time.perf_counter() - loop_start
                instant = 1.0 / elapsed if elapsed > 0 else 0.0
                smoothed_fps = instant if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant
                frame_index += 1
                if frame_index % 50 == 0:
                    progress = f"{100.0 * frame_index / total:.0f}%" if total > 0 else f"{frame_index} frames"
                    print(f"[RUN] {progress}  processing FPS: {smoothed_fps:.1f}  "
                          f"alerts: {self.alert_manager.alert_count}")
        except KeyboardInterrupt:
            print("\n[RUN] Interrupted.")
        finally:
            capture.release()
            writer.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            self.event_logger.finalize(self.last_frame_index, self.last_timestamp)

        total_time = time.perf_counter() - started
        print("-" * 64)
        print(f"[DONE] {frame_index} frames in {total_time:.1f} s "
              f"({frame_index / total_time if total_time > 0 else 0:.1f} FPS average)")
        print(f"[DONE] Output video : {self.output_path}")
        print(f"[DONE] Event log    : {cfg.EVENTS_JSON}  ({len(self.event_logger.events)} events)")
        print(f"[DONE] Screenshots  : {cfg.EVENT_SCREENSHOT_DIR}")
        return 0

    # ------------------------------------------------------------------ per frame
    def _process_frame(self, frame, frame_index: int, processing_fps: float):
        timestamp = frame_index / self.fps
        self.last_frame_index, self.last_timestamp = frame_index, timestamp

        detections = self.detector.detect_and_track(frame)
        self.crowd.update(detections, timestamp)               # light data for ALL tracked people
        selected = self.track_manager.select_detections(detections)   # only people on the escalator
        pose_map = self.pose_estimator.estimate(frame, selected, self.pose_crop) if selected else {}
        active, removed = self.track_manager.update(selected, pose_map, frame_index, timestamp)

        for person in removed:
            self._close_person_events(person, frame_index, timestamp)

        new_alerts = []
        for person in active:
            person.approach_speed = self.crowd.approach_speed(person.track_id)
            try:
                result = self.analyzer.analyze(person, timestamp)
                new_alerts += self._sync_person_events(person, result, frame_index, timestamp)
            except Exception:                                   # noqa: BLE001 - never crash the video loop
                self._error_count += 1
                if self._error_count <= 3:
                    print(f"[WARN] Analysis error for person #{person.track_id} (continuing):")
                    traceback.print_exc()

        fallen_ids = [p.track_id for p in active if p.state in DANGER_STATES]
        incidents = self.crowd.detect_incidents(fallen_ids, timestamp)
        new_alerts += self._sync_crowd_events(incidents, frame_index, timestamp)

        # ---- drawing ----
        viz = self.visualizer
        viz.draw_roi(frame)
        for person in active:
            viz.draw_person(frame, person)
        for incident in incidents:
            viz.draw_crowd_incident(frame, incident)
        status, status_color = self._system_status(active, incidents)
        viz.draw_hud(frame, format_timestamp(timestamp), processing_fps, len(active),
                     self.alert_manager.alert_count, status, status_color)
        banner = self.alert_manager.current_banner(timestamp)
        if banner is not None:
            viz.draw_banner(frame, banner, frame_index)
        viz.draw_footer(frame)

        # one representative (annotated) screenshot per NEW event
        for alert, event in new_alerts:
            self.event_logger.save_screenshot(event, frame)
            stamp = format_timestamp(timestamp)
            if alert is not None:
                print(f"[ALERT] {stamp}  {alert.title}  person #{alert.track_id}  "
                      f"score {alert.confidence:.2f}")
            else:  # event logged, but the alert was suppressed by the cooldown
                print(f"[EVENT] {stamp}  {event['event_type']}  person #{event['track_id']}  "
                      f"(alert suppressed by cooldown)")
        return frame

        # ------------------------------------------------------------------ events

    @staticmethod
    def _num(value, digits: int = 3):
        """Finite number -> rounded float, otherwise None (NaN is not valid JSON)."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return round(value, digits) if math.isfinite(value) else None

    def _desired_events(self, person, result, timestamp) -> dict:
        """Which event types should currently be open for this person? {type: confidence}"""
        S = BehaviorState
        if person.state == S.FALLING:
            return {"fall": result.fall.confidence}
        if person.state == S.PERSON_DOWN:
            return {"person_down": result.confidence}
        if person.state == S.SITTING:
            return {"sitting": result.sitting.confidence}
        if (person.state == S.SUSPICIOUS
                and timestamp - person.state_since >= cfg.UNUSUAL_EVENT_MIN_SEC):
            return {"unusual_behavior": result.confidence}
        return {}

    def _event_details(self, person, result, event_type: str) -> dict:
        """Extra explainability fields stored in events.json."""
        details = {
            "behavior": person.behavior,
            "horizontal_score": self._num(person.horizontal_score),
            "torso_angle_deg": self._num(person.torso_angle_smooth, 1),
            "movement_direction": person.movement_direction,
            "bbox": [round(float(v), 1) for v in person.bbox],
            "pose_available": bool(person.has_pose),
        }
        if event_type in ("fall", "person_down"):
            details["fall_score"] = self._num(result.fall.score)
            details["fall_evidence"] = result.fall.evidence
            details["fall_observed"] = bool(person.had_fall)
        elif event_type == "sitting":
            details["sitting_signals"] = result.sitting.signals
        elif event_type == "unusual_behavior":
            details["category"] = person.suspicion_category
            details["reasons"] = list(person.suspicion_reasons)
        return details

    def _sync_person_events(self, person, result, frame_index: int, timestamp: float) -> list:
        """Open, update and close the events of one person. Returns [(alert_or_None, event)]
        for NEW events only (used for screenshots and console messages)."""
        new_items = []
        desired = self._desired_events(person, result, timestamp)

        for event_type, confidence in desired.items():
            event = person.active_events.get(event_type)
            if event is None:
                event, is_new = self.event_logger.open_event(
                    event_type, person.track_id, confidence, frame_index, timestamp,
                    self._event_details(person, result, event_type))
                person.active_events[event_type] = event
                if is_new:
                    alert = self.alert_manager.try_raise(
                        event_type, person.track_id, confidence, timestamp, frame_index)
                    if alert is not None:
                        person.last_alert_time = timestamp
                    new_items.append((alert, event))
            else:
                self.event_logger.update_confidence(event, confidence)  # keep the peak

        for event_type in [t for t in person.active_events if t not in desired]:
            self.event_logger.close_event(person.active_events.pop(event_type), frame_index, timestamp)
        return new_items

    def _close_person_events(self, person, frame_index: int, timestamp: float) -> None:
        for event in list(person.active_events.values()):
            self.event_logger.close_event(event, frame_index, timestamp)
        person.active_events.clear()

    def _sync_crowd_events(self, incidents, frame_index: int, timestamp: float) -> list:
        new_items = []
        current_ids = set()
        for incident in incidents:
            current_ids.add(incident.track_id)
            event = self._crowd_events.get(incident.track_id)
            if event is None:
                details = {"near_track_ids": [int(i) for i in incident.near_ids],
                           "arriving_track_ids": [int(i) for i in incident.arriving_ids],
                           "note": "automated visual warning; does not prove an emergency"}
                event, is_new = self.event_logger.open_event(
                    "crowd_incident", incident.track_id, incident.confidence,
                    frame_index, timestamp, details)
                self._crowd_events[incident.track_id] = event
                if is_new:
                    alert = self.alert_manager.try_raise(
                        "crowd_incident", incident.track_id, incident.confidence, timestamp, frame_index)
                    new_items.append((alert, event))
            else:
                self.event_logger.update_confidence(event, incident.confidence)
        for track_id in [t for t in self._crowd_events if t not in current_ids]:
            self.event_logger.close_event(self._crowd_events.pop(track_id), frame_index, timestamp)
        return new_items

    # ------------------------------------------------------------------ status text
    @staticmethod
    def _system_status(active, incidents):
        if incidents:
            return "ALERT - POSSIBLE CROWD INCIDENT", RED
        if any(p.state in DANGER_STATES for p in active):
            return "ALERT - PERSON DOWN / FALL", RED
        if any(p.state in (BehaviorState.SUSPICIOUS, BehaviorState.SITTING) for p in active):
            return "WARNING - REVIEW", ORANGE
        return "NORMAL", GREEN


def parse_arguments():
    parser = argparse.ArgumentParser(description="Escalator Safety Monitor (automated visual warning system)")
    parser.add_argument("--input", default=str(cfg.INPUT_VIDEO), help="input video path")
    parser.add_argument("--output", default=str(cfg.OUTPUT_VIDEO), help="output video path")
    parser.add_argument("--no-preview", action="store_true", help="do not open the live preview window")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    monitor = EscalatorSafetyMonitor(
        Path(args.input), Path(args.output),
        show_preview=cfg.SHOW_PREVIEW and not args.no_preview)
    return monitor.run()


if __name__ == "__main__":
    raise SystemExit(main())
