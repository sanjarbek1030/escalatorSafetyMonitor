"""
behavior_analyzer.py - combines the detectors and runs the per-person state machine.

Contains:
  UnusualBehaviorAnalyzer : suspicion score + category (heuristic)
  BehaviorStateMachine    : NORMAL / SUSPICIOUS / SITTING / FALLING / PERSON_DOWN / RECOVERING
  BehaviorAnalyzer        : the entry point used by main.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config as cfg
from fall_detector import FallAssessment, FallDetector
from person_state import BehaviorState
from sitting_detector import SittingAssessment, SittingDetector
from utils import clip, is_finite, ramp


@dataclass
class UnusualAssessment:
    score: float = 0.0
    category: str = None
    reasons: list = field(default_factory=list)


class UnusualBehaviorAnalyzer:
    CATEGORIES = {"reverse": "REVERSE_MOVEMENT",
                  "abnormal_torso": "UNUSUAL_POSTURE",
                  "crouching": "UNUSUAL_POSTURE",
                  "crawling": "UNUSUAL_POSTURE"}

    def evaluate(self, person, now, fall: FallAssessment, sitting: SittingAssessment) -> UnusualAssessment:
        result = UnusualAssessment()
        if not person.warmed_up:
            return result
        w = cfg.UNUSUAL_WEIGHTS
        points = {}

        # Reverse movement: moving AGAINST the escalator direction for a while.
        reverse_active = (person.movement_direction == "against" and person.speed >= cfg.REVERSE_MIN_SPEED)
        held = person.timer("reverse").update(reverse_active, now)
        points["reverse"] = w["reverse"] if held >= cfg.REVERSE_MIN_SEC else 0.0

        # Sudden acceleration (recent peak).
        recent_a = person.window(person.accel_history, 0.5)
        peak_a = max(recent_a) if recent_a else 0.0
        points["sudden_acceleration"] = w["sudden_acceleration"] * ramp(peak_a, *cfg.SUSPICIOUS_ACCEL_RANGE)

        # Abnormal (leaning) torso that is not yet horizontal.
        torso = 0.0
        if is_finite(person.torso_angle_smooth) and person.horizontal_score < cfg.HORIZONTAL_SCORE_MIN:
            torso = ramp(person.torso_angle_smooth, *cfg.ABNORMAL_TORSO_RANGE)
        points["abnormal_torso"] = w["abnormal_torso"] * torso

        # Prolonged crouching: knees strongly bent + body noticeably shorter, but not sitting.
        heights = person.window(person.height_history, cfg.HISTORY_SECONDS)
        height_ratio = 1.0
        if len(heights) >= 5:
            reference = float(np.percentile(heights, 90))
            if reference > 1e-6:
                height_ratio = heights[-1] / reference
        crouch_active = (is_finite(person.mean_knee_angle) and person.mean_knee_angle < cfg.CROUCH_KNEE_MAX
                         and height_ratio < cfg.CROUCH_HEIGHT_RATIO
                         and sitting.score < cfg.SIT_SCORE_THRESHOLD)
        held = person.timer("crouch").update(crouch_active, now)
        points["crouching"] = w["crouching"] if held >= cfg.CROUCH_MIN_SEC else 0.0

        # Crawling-like: low/horizontal body that is actively moving its limbs.
        crawl_active = (fall.horizontal_score >= cfg.CRAWL_HORIZONTAL_MIN
                        and person.internal_motion >= cfg.CRAWL_INTERNAL_MOTION_MIN)
        held = person.timer("crawl").update(crawl_active, now)
        points["crawling"] = w["crawling"] if held >= cfg.CRAWL_MIN_SEC else 0.0

        # Long stationary period (contributing feature only; weight alone < threshold).
        held = person.timer("stationary").update(person.movement_direction == "stationary", now)
        points["long_stationary"] = w["long_stationary"] if held >= cfg.LONG_STATIONARY_SEC else 0.0

        # Sudden movement toward another person (from CrowdAnalyzer).
        points["approaching_person"] = w["approaching_person"] * ramp(person.approach_speed, *cfg.APPROACH_SPEED_RANGE)

        # Rapid change of body orientation (degrees per second).
        angles = [a for a in person.window(person.torso_angle_history, cfg.ROTATION_RATE_WINDOW_SEC) if is_finite(a)]
        rate = (max(angles) - min(angles)) / cfg.ROTATION_RATE_WINDOW_SEC if len(angles) >= 2 else 0.0
        points["rapid_rotation"] = w["rapid_rotation"] * ramp(rate, *cfg.ROTATION_RATE_RANGE)

        result.score = float(min(1.0, sum(points.values())))
        ordered = sorted(points.items(), key=lambda kv: kv[1], reverse=True)
        result.reasons = [name for name, value in ordered if value >= 0.05]
        if ordered and ordered[0][1] >= 0.05:
            result.category = self.CATEGORIES.get(ordered[0][0], "SUSPICIOUS")
        return result


class BehaviorStateMachine:
    """
    Per-person state machine. Every transition needs a condition to hold for a minimum
    time (ConditionTimer), which prevents NORMAL -> FALL -> NORMAL flicker.

        NORMAL <-> SUSPICIOUS,  NORMAL/SUSPICIOUS <-> SITTING
        any of those -> FALLING -> PERSON_DOWN -> RECOVERING -> NORMAL
    """

    def step(self, person, fall, sitting, unusual, now):
        S = BehaviorState
        fall_condition = (fall.score >= cfg.FALL_SCORE_THRESHOLD
                          and fall.horizontal_score >= cfg.FALL_MIN_HORIZONTAL_SCORE)
        fall_hold = person.timer("fall_confirm").update(fall_condition, now)
        upright_hold = person.timer("upright").update(
            fall.horizontal_score < cfg.RECOVERED_HORIZONTAL_SCORE, now)
        suspicious_hold = person.timer("suspicious").update(unusual.score >= cfg.SUSPICIOUS_SCORE_ENTER, now)
        calm_hold = person.timer("calm").update(unusual.score < cfg.SUSPICIOUS_SCORE_EXIT, now)
        not_sitting_hold = person.timer("not_sitting").update(sitting.score < cfg.SIT_SCORE_EXIT_THRESHOLD, now)
        down = fall.down_duration

        old = person.state
        new = old
        time_in_state = now - person.state_since

        if old in (S.NORMAL, S.SUSPICIOUS, S.SITTING):
            if fall_hold >= cfg.FALL_CONFIRM_SEC:
                new = S.FALLING
            elif down >= cfg.PERSON_DOWN_NO_FALL_DURATION:
                new = S.PERSON_DOWN                 # person is down but we never saw the fall
            elif old != S.SITTING and sitting.confirmed:
                new = S.SITTING
            elif old == S.SITTING and not_sitting_hold >= cfg.SIT_EXIT_SEC:
                new = S.NORMAL
            elif old == S.NORMAL and suspicious_hold >= cfg.SUSPICIOUS_ENTER_SEC:
                new = S.SUSPICIOUS
            elif old == S.SUSPICIOUS and calm_hold >= cfg.SUSPICIOUS_EXIT_SEC:
                new = S.NORMAL
        elif old == S.FALLING:
            if down >= cfg.PERSON_DOWN_DURATION:
                new = S.PERSON_DOWN
            elif upright_hold >= cfg.RECOVER_CONFIRM_SEC:
                new = S.RECOVERING
            elif time_in_state >= cfg.FALLING_MAX_SEC:
                new = S.PERSON_DOWN if fall.is_horizontal else S.RECOVERING
        elif old == S.PERSON_DOWN:
            if upright_hold >= cfg.RECOVER_CONFIRM_SEC:
                new = S.RECOVERING
        elif old == S.RECOVERING:
            if down >= cfg.PERSON_DOWN_DURATION:
                new = S.PERSON_DOWN
            elif fall_hold >= cfg.FALL_CONFIRM_SEC:
                new = S.FALLING
            elif time_in_state >= cfg.RECOVERY_DURATION_SEC:
                new = S.NORMAL

        if new == old:
            return None
        person.state = new
        person.state_since = now
        if new == S.FALLING:
            person.had_fall = True
        if new == S.NORMAL:
            person.had_fall = False
            person.event_start_time = None
        elif old == S.NORMAL or person.event_start_time is None:
            person.event_start_time = now
        for name in ("fall_confirm", "suspicious", "calm", "not_sitting", "upright"):
            person.timer(name).reset()             # avoid instant re-triggering after a transition
        return (old, new)


@dataclass
class BehaviorResult:
    state: BehaviorState
    label: str
    confidence: float
    transition: tuple
    fall: FallAssessment
    sitting: SittingAssessment
    unusual: UnusualAssessment


class BehaviorAnalyzer:
    def __init__(self):
        self.fall_detector = FallDetector()
        self.sitting_detector = SittingDetector()
        self.unusual_analyzer = UnusualBehaviorAnalyzer()
        self.state_machine = BehaviorStateMachine()

    def analyze(self, person, now: float) -> BehaviorResult:
        fall = self.fall_detector.evaluate(person, now)
        sitting = self.sitting_detector.evaluate(person, now)
        unusual = self.unusual_analyzer.evaluate(person, now, fall, sitting)

        if not person.warmed_up:
            person.behavior = "ANALYZING"
            person.behavior_confidence = 0.0
            return BehaviorResult(person.state, "ANALYZING", 0.0, None, fall, sitting, unusual)

        transition = self.state_machine.step(person, fall, sitting, unusual, now)
        state = person.state
        person.suspicion_category = unusual.category if state == BehaviorState.SUSPICIOUS else None
        person.suspicion_reasons = unusual.reasons if state == BehaviorState.SUSPICIOUS else []

        label = state.value
        if state == BehaviorState.SUSPICIOUS and unusual.category in ("REVERSE_MOVEMENT", "UNUSUAL_POSTURE"):
            label = unusual.category
        confidence = self._confidence(person, state, fall, sitting, unusual)
        person.behavior = label
        person.behavior_confidence = confidence
        person.record_behavior(label)
        return BehaviorResult(state, label, confidence, transition, fall, sitting, unusual)

    @staticmethod
    def _confidence(person, state, fall, sitting, unusual) -> float:
        """Composite behavior score in 0..1. NOT a calibrated probability."""
        S = BehaviorState
        if state == S.FALLING:
            return fall.confidence
        if state == S.PERSON_DOWN:
            needed = cfg.PERSON_DOWN_DURATION if person.had_fall else cfg.PERSON_DOWN_NO_FALL_DURATION
            persistence = clip(fall.down_duration / (2.0 * needed), 0.0, 1.0)
            return clip(0.35 * fall.horizontal_score
                        + 0.25 * (1.0 if fall.is_still else 0.3)
                        + 0.25 * persistence
                        + 0.15 * (1.0 if person.had_fall else 0.0), 0.0, 1.0)
        if state == S.SITTING:
            return clip(sitting.confidence, 0.0, 1.0)
        if state == S.SUSPICIOUS:
            return clip(unusual.score, 0.0, 1.0)
        if state == S.RECOVERING:
            return clip(1.0 - fall.horizontal_score, 0.0, 1.0)
        risk = max(unusual.score,
                   0.8 * clip(fall.score / cfg.FALL_SCORE_THRESHOLD, 0.0, 1.0),
                   0.6 * sitting.score)
        return clip(1.0 - risk, 0.0, 1.0)


