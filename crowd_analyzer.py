"""
crowd_analyzer.py - simple pairwise proximity analysis.

Keeps a light position history for EVERY tracked person (also outside the ROI), computes
pairwise distances in body-lengths and reports:
  * approach speed of each person toward anyone else (used as an 'unusual' feature)
  * POSSIBLE CROWD INCIDENT: several people suddenly gather around a fallen person.
This is an automated visual warning only. It does not prove that an emergency exists.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

import config as cfg
from utils import ConditionTimer, clip


@dataclass
class CrowdIncident:
    track_id: int          # the fallen person the crowd gathers around
    near_ids: list
    arriving_ids: list
    confidence: float      # composite score, not a calibrated probability
    center: tuple
    radius: float


class CrowdAnalyzer:
    def __init__(self, fps: float):
        self.max_history = max(30, int(fps * 3))
        self.tracks = {}          # id -> deque of (time, cx, cy, scale)
        self.last_seen = {}
        self._approach = {}
        self._index = {}
        self._dist_now = np.zeros((0, 0))
        self._dist_past = np.zeros((0, 0))
        self._scales = np.zeros(0)
        self._centers = np.zeros((0, 2))
        self._timers = {}

    def update(self, detections, now: float) -> None:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            entry = (now, (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1, 1.0))
            self.tracks.setdefault(det.track_id, deque(maxlen=self.max_history)).append(entry)
            self.last_seen[det.track_id] = now
        for track_id in [t for t, seen in self.last_seen.items() if now - seen > cfg.CROWD_TRACK_TIMEOUT_SEC]:
            self.tracks.pop(track_id, None)
            self.last_seen.pop(track_id, None)
        self._compute_pairs([d.track_id for d in detections], now)

    def _compute_pairs(self, ids, now):
        n = len(ids)
        self._approach = {}
        self._index = {track_id: i for i, track_id in enumerate(ids)}
        self._dist_now = np.zeros((n, n))
        self._dist_past = np.full((n, n), np.nan)
        self._scales = np.zeros(n)
        self._centers = np.zeros((n, 2))
        if n == 0:
            return
        now_pos = np.zeros((n, 2))
        past_pos = np.full((n, 2), np.nan)
        for i, track_id in enumerate(ids):
            history = self.tracks[track_id]
            _, cx, cy, scale = history[-1]
            now_pos[i] = (cx, cy)
            self._scales[i] = scale
            target = now - cfg.CROWD_LOOKBACK_SEC
            for entry in reversed(history):
                if entry[0] <= target:
                    past_pos[i] = (entry[1], entry[2])
                    break
        self._centers = now_pos
        if n < 2:
            return
        self._dist_now = np.linalg.norm(now_pos[:, None, :] - now_pos[None, :, :], axis=2)
        self._dist_past = np.linalg.norm(past_pos[:, None, :] - past_pos[None, :, :], axis=2)  # NaN if unknown
        pair_scale = (self._scales[:, None] + self._scales[None, :]) / 2.0
        closing = (self._dist_past - self._dist_now) / pair_scale / cfg.CROWD_LOOKBACK_SEC     # BH/s
        np.fill_diagonal(closing, np.nan)
        closing = np.where(np.isfinite(closing), closing, -np.inf)
        best = closing.max(axis=1)
        for track_id, i in self._index.items():
            self._approach[track_id] = float(max(0.0, best[i])) if np.isfinite(best[i]) else 0.0

    def approach_speed(self, track_id: int) -> float:
        return self._approach.get(track_id, 0.0)

    def detect_incidents(self, fallen_ids, now: float) -> list:
        incidents = []
        for stale in [t for t in self._timers if t not in fallen_ids]:
            del self._timers[stale]
        for track_id in fallen_ids:
            i = self._index.get(track_id)
            if i is None or len(self._scales) < 2:
                continue
            radius = cfg.CROWD_RADIUS_BODY_SCALES * self._scales[i]
            others = [j for j in range(len(self._scales)) if j != i]
            near = [j for j in others if self._dist_now[i, j] <= radius]
            arriving = [j for j in near if np.isfinite(self._dist_past[i, j])
                        and self._dist_past[i, j] > radius * cfg.CROWD_ARRIVAL_FACTOR]
            condition = len(near) >= cfg.CROWD_MIN_NEAR and len(arriving) >= cfg.CROWD_MIN_ARRIVING
            timer = self._timers.setdefault(track_id, ConditionTimer(cfg.CROWD_GRACE_SEC))
            held = timer.update(condition, now)
            if held >= cfg.CROWD_CONFIRM_SEC:
                id_of = {v: k for k, v in self._index.items()}
                confidence = clip(0.5 * min(1.0, len(near) / (2.0 * cfg.CROWD_MIN_NEAR))
                                  + 0.5 * min(1.0, len(arriving) / (2.0 * cfg.CROWD_MIN_ARRIVING)), 0.0, 1.0)
                incidents.append(CrowdIncident(
                    track_id=track_id,
                    near_ids=[id_of[j] for j in near],
                    arriving_ids=[id_of[j] for j in arriving],
                    confidence=confidence,
                    center=(float(self._centers[i][0]), float(self._centers[i][1])),
                    radius=float(radius)))
        return incidents

