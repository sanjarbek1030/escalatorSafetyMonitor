"""alert_manager.py - decides WHEN an alert may be raised (cooldown) and tracks the banner."""
from __future__ import annotations

from dataclasses import dataclass

import config as cfg

ALERT_TITLES = {
    "fall": "FALL DETECTED",
    "person_down": "PERSON DOWN",
    "crowd_incident": "POSSIBLE CROWD INCIDENT",
    "sitting": "PERSON SITTING",
    "unusual_behavior": "UNUSUAL BEHAVIOR",
}


@dataclass
class Alert:
    event_type: str
    track_id: int
    confidence: float
    timestamp: float
    frame_index: int
    title: str
    show_banner: bool


class AlertManager:
    def __init__(self):
        self.alert_count = 0
        self._last_alert_time = {}       # (event_type, track_id) -> video time
        self._banner_alert = None

    def try_raise(self, event_type, track_id, confidence, timestamp, frame_index):
        """Returns an Alert, or None if this (type, person) alerted less than ALERT_COOLDOWN ago."""
        key = (event_type, track_id)
        last = self._last_alert_time.get(key)
        if last is not None and timestamp - last < cfg.ALERT_COOLDOWN:
            return None
        self._last_alert_time[key] = timestamp
        self.alert_count += 1
        alert = Alert(event_type, track_id, confidence, timestamp, frame_index,
                      ALERT_TITLES.get(event_type, event_type.upper()),
                      event_type in cfg.BANNER_EVENT_TYPES)
        if alert.show_banner:
            self._banner_alert = alert
        return alert

    def current_banner(self, now: float):
        if self._banner_alert is not None and now - self._banner_alert.timestamp <= cfg.BANNER_DURATION_SEC:
            return self._banner_alert
        return None

