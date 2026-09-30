"""utils.py - small math / time helpers shared by several modules."""
from __future__ import annotations

import math

import numpy as np


def clip(value, low, high):
    return max(low, min(high, value))


def is_finite(value) -> bool:
    return value is not None and math.isfinite(value)


def ramp(value, low, high) -> float:
    """0 at or below `low`, 1 at or above `high`, linear in between. NaN/None -> 0."""
    if not is_finite(value):
        return 0.0
    if high <= low:
        return 1.0 if value >= high else 0.0
    return clip((value - low) / (high - low), 0.0, 1.0)


def trapezoid(value, zero_low, full_low, full_high, zero_high) -> float:
    """Membership that rises zero_low->full_low, stays 1 until full_high, falls to zero_high."""
    if not is_finite(value) or value <= zero_low or value >= zero_high:
        return 0.0
    if value < full_low:
        return (value - zero_low) / (full_low - zero_low)
    if value > full_high:
        return (zero_high - value) / (zero_high - full_high)
    return 1.0


def iou(box_a, box_b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cosine_similarity(vec_a, vec_b) -> float:
    """cos(angle) between two vectors: +1 same direction, 0 perpendicular, -1 opposite."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def format_timestamp(seconds: float) -> str:
    """Seconds -> 'HH:MM:SS.mmm'."""
    total_ms = int(round(max(0.0, seconds) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class ConditionTimer:
    """
    Measures how long a boolean condition has been continuously true, while tolerating
    short dropouts (`grace` seconds). This is the basic tool for "temporal confirmation".

    update(active, now) returns the current continuous duration in seconds.
    """

    def __init__(self, grace: float):
        self.grace = grace
        self.start = None
        self.last_true = None

    def update(self, active: bool, now: float) -> float:
        if active:
            if self.start is None or now - self.last_true > self.grace:
                self.start = now
            self.last_true = now
            return now - self.start
        if self.start is not None and now - self.last_true <= self.grace:
            return self.last_true - self.start      # inside grace period: hold the duration
        self.start = None
        self.last_true = None
        return 0.0

    def reset(self) -> None:
        self.start = None
        self.last_true = None
