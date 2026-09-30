"""event_logger.py - writes output/events.json and one representative screenshot per event."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import config as cfg
from utils import format_timestamp

DISCLAIMER = ("Automated visual warning system output. Heuristic rules can produce false positives "
              "and false negatives. Every entry requires human review.")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class EventLogger:
    def __init__(self, json_path, screenshot_dir, video_name: str, fps: float):
        self.json_path = Path(json_path)
        self.screenshot_dir = Path(screenshot_dir)
        self.video_name = video_name
        self.fps = fps
        self.events = []
        self._open = {}                 # (track_id, event_type) -> event
        self._recent_closed = {}        # (track_id, event_type) -> (event, closed_at_time)
        self._next_id = 1
        self.frames_processed = 0
        try:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            if cfg.CLEAR_OLD_SCREENSHOTS:
                for old in self.screenshot_dir.glob("*.jpg"):
                    old.unlink()
        except OSError as exc:
            print(f"[WARN] Could not prepare output folders: {exc}")
        self.save_json()

    def open_event(self, event_type, track_id, confidence, frame_index, timestamp, details=None):
        """Returns (event, is_new). Re-opens a very recent event of the same person and type
        instead of creating a duplicate."""
        key = (track_id, event_type)
        if key in self._open:
            return self._open[key], False
        recent = self._recent_closed.get(key)
        if recent is not None and timestamp - recent[1] <= cfg.EVENT_MERGE_SECONDS:
            event = recent[0]
            event.update(status="open", end_frame=None, end_timestamp=None, duration_seconds=None)
            event["reopen_count"] = event.get("reopen_count", 0) + 1
            del self._recent_closed[key]
            self._open[key] = event
            self.save_json()
            return event, False
        event = {
            "event_id": self._next_id,
            "track_id": int(track_id),
            "event_type": event_type,
            "confidence": round(float(confidence), 3),
            "start_frame": int(frame_index),
            "end_frame": None,
            "timestamp": format_timestamp(timestamp),
            "end_timestamp": None,
            "duration_seconds": None,
            "status": "open",
            "screenshot": None,
            "reopen_count": 0,
            "details": details or {},
            "_start_time": float(timestamp),
        }
        self._next_id += 1
        self.events.append(event)
        self._open[key] = event
        self.save_json()
        return event, True

    def update_confidence(self, event, confidence: float) -> None:
        event["confidence"] = round(max(event["confidence"], float(confidence)), 3)   # keep the peak

    def close_event(self, event, frame_index, timestamp) -> None:
        if event.get("status") != "open":
            return
        event["end_frame"] = int(frame_index)
        event["end_timestamp"] = format_timestamp(timestamp)
        event["duration_seconds"] = round(max(0.0, timestamp - event["_start_time"]), 3)
        event["status"] = "closed"
        key = (event["track_id"], event["event_type"])
        self._open.pop(key, None)
        self._recent_closed[key] = (event, timestamp)
        self.save_json()

    def save_screenshot(self, event, frame) -> None:
        name = f"{event['event_type']}_{event['event_id']:04d}.jpg"
        path = self.screenshot_dir / name
        try:
            ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except cv2.error:
            ok = False
        if ok:
            event["screenshot"] = f"{self.screenshot_dir.name}/{name}"
            self.save_json()
        else:
            print(f"[WARN] Could not write screenshot {path}")

    def finalize(self, last_frame_index, last_timestamp) -> None:
        for event in list(self._open.values()):
            self.close_event(event, last_frame_index, last_timestamp)
        self.frames_processed = last_frame_index + 1
        self.save_json()

    def save_json(self) -> None:
        public_events = [{k: v for k, v in e.items() if not k.startswith("_")} for e in self.events]
        payload = {
            "video": self.video_name,
            "fps": self.fps,
            "frames_processed": self.frames_processed,
            "event_count": len(public_events),
            "disclaimer": DISCLAIMER,
            "confidence_note": "confidence is a composite heuristic score, not a calibrated probability",
            "events": public_events,
        }
        try:
            with open(self.json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=_json_default)
        except OSError as exc:
            print(f"[WARN] Could not write {self.json_path}: {exc}")

