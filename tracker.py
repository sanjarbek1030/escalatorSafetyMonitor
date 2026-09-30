"""
tracker.py - owns the PersonState object of every tracked person.

The tracking ALGORITHM (ByteTrack) runs inside detector.py. This class handles the
lifecycle of our per-person state: create, update, and expire it.
"""
from __future__ import annotations

import config as cfg
from person_state import DANGER_STATES, PersonState


class TrackManager:
    def __init__(self, escalator, fps: float):
        self.escalator = escalator
        self.fps = fps
        self.persons = {}                 # track_id -> PersonState

    def select_detections(self, detections):
        """Keep only people on the escalator (plus people already in a danger state)."""
        selected = []
        for det in detections:
            existing = self.persons.get(det.track_id)
            already_in_danger = existing is not None and existing.state in DANGER_STATES
            if already_in_danger or self.escalator.contains_person(det.bbox):
                selected.append(det)
        return selected

    def update(self, detections, pose_map, frame_index: int, timestamp: float):
        """Returns (active_persons, removed_persons)."""
        active = []
        for det in detections:
            person = self.persons.get(det.track_id)
            if person is None:
                person = PersonState(det.track_id, self.fps, timestamp)
                self.persons[det.track_id] = person
            pose = pose_map.get(det.track_id)
            person.update(
                bbox=det.bbox,
                keypoints_xy=pose.xy if pose is not None else None,
                keypoints_conf=pose.conf if pose is not None else None,
                frame_index=frame_index,
                timestamp=timestamp,
                escalator_direction=self.escalator.direction,
            )
            active.append(person)

        removed = []
        for track_id, person in list(self.persons.items()):
            timeout = (cfg.DANGER_TRACK_TIMEOUT_SEC if person.state in DANGER_STATES
                       else cfg.LOST_TRACK_TIMEOUT_SEC)
            if timestamp - person.last_seen_time > timeout:
                removed.append(person)
                del self.persons[track_id]
        return active, removed
