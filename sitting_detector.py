"""
sitting_detector.py - is the person sitting on the escalator?

Heuristic multi-signal score (knee bend, low hips, folded legs, compact box, upright
torso) that must persist for SITTING_DURATION seconds. Requires pose keypoints:
a bounding box alone can never confirm sitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config as cfg
from utils import is_finite, ramp, trapezoid


@dataclass
class SittingAssessment:
    score: float = 0.0
    confirmed: bool = False
    confidence: float = 0.0
    duration: float = 0.0
    signals: dict = field(default_factory=dict)


class SittingDetector:
    def evaluate(self, person, now: float) -> SittingAssessment:
        result = SittingAssessment()
        w = cfg.SIT_WEIGHTS
        signals = []       # (name, weight, value 0..1)

        if is_finite(person.mean_knee_angle):
            signals.append(("knee", w["knee"], trapezoid(person.mean_knee_angle, *cfg.SIT_KNEE_TRAPEZOID)))
        if is_finite(person.hip_ratio):
            signals.append(("hip_low", w["hip_low"], ramp(person.hip_ratio, *cfg.SIT_HIP_RATIO_RANGE)))
        if is_finite(person.leg_extent_ratio):
            signals.append(("legs_folded", w["legs_folded"],
                            1.0 - ramp(person.leg_extent_ratio, *cfg.SIT_LEG_EXTENT_RANGE)))
        signals.append(("compact_box", w["compact_box"], trapezoid(person.aspect_ratio, *cfg.SIT_ASPECT_TRAPEZOID)))
        if is_finite(person.torso_angle_smooth):
            signals.append(("torso_upright", w["torso_upright"],
                            1.0 - ramp(person.torso_angle_smooth, *cfg.SIT_TORSO_RANGE)))

        available = sum(weight for _, weight, _ in signals)
        score = 0.0
        if (person.warmed_up and available >= cfg.SIT_MIN_AVAILABLE_WEIGHT
                and person.horizontal_score < cfg.SIT_MAX_HORIZONTAL_SCORE):
            score = sum(weight * value for _, weight, value in signals) / available
        result.score = float(score)
        result.signals = {name: round(float(weight * value), 3) for name, weight, value in signals}

        # Temporal persistence (with a small dropout tolerance from ConditionTimer)
        result.duration = person.timer("sitting_candidate").update(score >= cfg.SIT_SCORE_THRESHOLD, now)
        result.confirmed = result.duration >= cfg.SITTING_DURATION

        person.sit_score_history.append((now, score))
        recent = [s for t, s in person.sit_score_history if t >= now - cfg.SITTING_DURATION]
        result.confidence = float(np.mean(recent)) if recent else 0.0
        return result

