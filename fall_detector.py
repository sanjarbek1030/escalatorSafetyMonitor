"""
fall_detector.py - temporal fall scoring.

This is a HEURISTIC rule-based scorer, not a learned model. It never decides from a
single frame: dynamic evidence comes from a rolling window and is peak-held for a few
seconds; 'horizontal' and 'transition' evidence look at the present and the recent past.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config as cfg
from utils import clip, is_finite, ramp


@dataclass
class FallAssessment:
    score: float = 0.0                  # weighted evidence sum (max = sum of FALL_WEIGHTS)
    confidence: float = 0.0             # composite score in 0..1 (NOT a calibrated probability)
    dynamic_score: float = 0.0
    horizontal_score: float = 0.0
    is_horizontal: bool = False
    is_still: bool = False
    down_duration: float = 0.0          # seconds of continuous 'horizontal and still'
    evidence: dict = field(default_factory=dict)


class FallDetector:
    def evaluate(self, person, now: float) -> FallAssessment:
        result = FallAssessment()
        result.horizontal_score = person.horizontal_score
        result.is_horizontal = person.horizontal_score >= cfg.HORIZONTAL_SCORE_MIN
        result.is_still = (person.internal_motion < cfg.STILL_INTERNAL_MOTION_MAX
                           and person.off_axis_speed < cfg.STILL_OFFAXIS_SPEED_MAX)

        down_candidate = person.warmed_up and result.is_horizontal and result.is_still
        result.down_duration = person.timer("down").update(down_candidate, now)
        if not person.warmed_up:
            return result

        weights, ranges = cfg.FALL_WEIGHTS, cfg.FALL_RANGES
        evidence = {}

        # Baseline = the person's own average velocity BEFORE the window. Subtracting it
        # removes the steady motion of the escalator, so we score only SUDDEN changes.
        older_v, recent_v = person.split_history(person.velocity_history, cfg.FALL_WINDOW_SEC)
        baseline = np.mean(older_v, axis=0) if len(older_v) >= 3 else np.zeros(2)

        # 1. velocity: peak deviation from the baseline velocity
        peak_speed = 0.0
        if recent_v:
            peak_speed = float(np.max(np.linalg.norm(np.asarray(recent_v) - baseline, axis=1)))
        evidence["velocity"] = weights["velocity"] * ramp(peak_speed, *ranges["velocity"])

        # 2. sudden vertical displacement (image y grows DOWNWARD), minus expected drift
        start = person.first_index_within(cfg.FALL_WINDOW_SEC)
        excess_drop = 0.0
        if start < len(person.timestamps) - 1:
            elapsed = person.timestamps[-1] - person.timestamps[start]
            drop = (person.center_history[-1][1] - person.center_history[start][1]) / person.body_scale
            excess_drop = drop - float(baseline[1]) * elapsed
        evidence["displacement"] = weights["displacement"] * ramp(excess_drop, *ranges["displacement"])

        # 3. acceleration
        recent_a = person.window(person.accel_history, cfg.FALL_WINDOW_SEC)
        peak_accel = max(recent_a) if recent_a else 0.0
        evidence["acceleration"] = weights["acceleration"] * ramp(peak_accel, *ranges["acceleration"])

        # 4. torso rotation (fallback: change of the horizontal score, scaled to ~degrees)
        angles = [a for a in person.window(person.torso_angle_history, cfg.FALL_WINDOW_SEC) if is_finite(a)]
        if len(angles) >= 2:
            rotation = max(angles) - min(angles)
        else:
            horiz = person.window(person.horizontal_history, cfg.FALL_WINDOW_SEC)
            rotation = 60.0 * (max(horiz) - min(horiz)) if len(horiz) >= 2 else 0.0
        evidence["rotation"] = weights["rotation"] * ramp(rotation, *ranges["rotation"])

        # 5. sudden decrease of body height (bbox), vs the 90th percentile of the last 2 s
        heights = person.window(person.height_history, cfg.FALL_LOOKBACK_SEC)
        height_drop = 0.0
        if len(heights) >= 5:
            reference = float(np.percentile(heights, 90))
            if reference > 1e-6:
                height_drop = 1.0 - heights[-1] / reference
        evidence["height_drop"] = weights["height_drop"] * ramp(height_drop, *ranges["height_drop"])

        # 6. pose collapse: the vertical spread of the skeleton shrinks
        extents = [e for e in person.window(person.pose_extent_history, cfg.FALL_LOOKBACK_SEC) if is_finite(e)]
        collapse = 0.0
        if len(extents) >= 5 and is_finite(person.pose_extent):
            reference = float(np.percentile(extents, 90))
            if reference > 1e-6:
                collapse = 1.0 - person.pose_extent / reference
        evidence["collapse"] = weights["collapse"] * ramp(collapse, *ranges["collapse"])

        dynamic = sum(evidence.values())

        # Peak-hold the dynamic evidence so it survives after the fall motion has left the window.
        if dynamic >= person.fall_memory_score or now - person.fall_memory_time > cfg.FALL_MEMORY_SEC:
            person.fall_memory_score = dynamic
            person.fall_memory_time = now
        remembered = person.fall_memory_score if now - person.fall_memory_time <= cfg.FALL_MEMORY_SEC else 0.0
        dynamic_effective = max(dynamic, remembered)

        # 7. current horizontal configuration
        horizontal_level = ramp(person.horizontal_score, *ranges["horizontal"])
        evidence["horizontal"] = weights["horizontal"] * horizontal_level

        # 8. posture transition: was upright recently AND is horizontal now
        recent_h = person.window(person.horizontal_history, cfg.FALL_LOOKBACK_SEC)
        was_upright = any(h <= cfg.UPRIGHT_SCORE_MAX for h in recent_h)
        evidence["transition"] = weights["transition"] * horizontal_level if was_upright else 0.0

        # 9. post-fall: stays down and still after a meaningful dynamic event
        post = 0.0
        if result.down_duration >= cfg.POST_FALL_STILL_SEC and dynamic_effective >= cfg.FALL_MEMORY_MIN_DYNAMIC:
            post = weights["post_fall"]
        evidence["post_fall"] = post

        result.dynamic_score = dynamic_effective
        result.score = dynamic_effective + evidence["horizontal"] + evidence["transition"] + post
        result.confidence = clip(result.score / cfg.FALL_SCORE_SATURATION, 0.0, 1.0)
        result.evidence = {k: round(float(v), 3) for k, v in evidence.items()}
        return result

