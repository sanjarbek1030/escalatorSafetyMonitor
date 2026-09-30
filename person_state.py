"""
person_state.py - everything we know about ONE tracked person.

PersonState stores current measurements, derived pose features, and bounded histories.
It only MEASURES. Deciding what the measurements mean is done by the detectors.
"""
from __future__ import annotations

import math
from collections import deque
from enum import Enum

import numpy as np

import config as cfg
from utils import ConditionTimer, clip, cosine_similarity, is_finite, ramp


class BehaviorState(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    SITTING = "SITTING"
    FALLING = "FALLING"
    PERSON_DOWN = "PERSON_DOWN"
    RECOVERING = "RECOVERING"


DANGER_STATES = (BehaviorState.FALLING, BehaviorState.PERSON_DOWN)

# COCO keypoint indices used by YOLO-Pose
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

SKELETON_EDGES = [
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER), (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]


# ---------------------------------------------------------------------------
# Small geometry helpers (all tolerate missing keypoints, which are NaN)
# ---------------------------------------------------------------------------
def _valid(point) -> bool:
    return point is not None and bool(np.all(np.isfinite(point)))


def midpoint(keypoints, index_a, index_b):
    """Midpoint of two keypoints; uses whichever one exists; None if neither does."""
    a, b = keypoints[index_a], keypoints[index_b]
    a_ok, b_ok = _valid(a), _valid(b)
    if a_ok and b_ok:
        return (a + b) / 2.0
    if a_ok:
        return a.copy()
    if b_ok:
        return b.copy()
    return None


def angle_at(point_a, point_b, point_c) -> float:
    """
    Angle in degrees at point_b between the segments b->a and b->c.
    For a knee: a=hip, b=knee, c=ankle. 180 = straight leg, ~90 = strongly bent.
    Uses the dot-product formula  cos(theta) = (u . v) / (|u| |v|).
    """
    if not (_valid(point_a) and _valid(point_b) and _valid(point_c)):
        return math.nan
    u = point_a - point_b
    v = point_c - point_b
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-6:
        return math.nan
    cos_theta = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def torso_geometry(keypoints):
    """
    Returns (torso_angle_deg, torso_length_px).
    torso_angle is the angle of the hip->shoulder line from the image VERTICAL:
    0 = upright, 90 = horizontal. atan2(|dx|, |dy|) ignores whether the head is up or
    down, so it stays robust when the body flips.
    """
    shoulders = midpoint(keypoints, L_SHOULDER, R_SHOULDER)
    hips = midpoint(keypoints, L_HIP, R_HIP)
    if shoulders is None or hips is None:
        return math.nan, math.nan
    vec = shoulders - hips
    length = float(np.linalg.norm(vec))
    if length < cfg.MIN_TORSO_PIXELS:
        return math.nan, math.nan
    return math.degrees(math.atan2(abs(vec[0]), abs(vec[1]))), length


class PersonState:
    def __init__(self, track_id: int, fps: float, timestamp: float):
        self.track_id = int(track_id)
        self.fps = float(fps)
        self.history_length = int(clip(round(cfg.HISTORY_SECONDS * self.fps),
                                       cfg.HISTORY_MIN_LENGTH, cfg.HISTORY_LENGTH))
        n = self.history_length

        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.last_frame_index = -1

        # --- current geometry ---
        self.bbox = (0.0, 0.0, 0.0, 0.0)
        self.bbox_width = 0.0
        self.bbox_height = 0.0
        self.aspect_ratio = 0.0                 # width / height
        self.center = np.zeros(2)
        self.previous_center = None
        self.foot_point = np.zeros(2)           # bottom-centre of the box (floor contact)
        self.body_scale = 1.0                   # smoothed max(w, h), used to normalise speeds

        # --- motion (speeds in body-lengths per second) ---
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.speed = 0.0                        # speed over DIRECTION_WINDOW_SEC
        self.along_speed = 0.0                  # signed speed along the escalator direction
        self.off_axis_speed = 0.0               # speed ACROSS the escalator direction
        self.direction_cosine = 0.0
        self.movement_direction = "stationary"  # stationary / with / against / lateral
        self.internal_motion = 0.0              # body-internal (pose) motion
        self.approach_speed = 0.0               # set from CrowdAnalyzer

        # --- pose ---
        self.has_pose = False
        self.keypoints_xy = None                # (17, 2), NaN where missing
        self.keypoints_conf = None
        self.torso_angle = math.nan
        self.torso_angle_smooth = math.nan
        self.knee_angles = (math.nan, math.nan)
        self.mean_knee_angle = math.nan
        self.hip_ratio = math.nan               # hip height inside bbox (0 top, 1 bottom)
        self.leg_extent_ratio = math.nan        # |ankle_y - hip_y| / torso length
        self.extent_aspect = math.nan           # keypoint x-extent / y-extent
        self.pose_extent = math.nan             # keypoint y-extent / body_scale
        self.horizontal_score = 0.0             # fused 0..1
        self.body_orientation = "upright"       # upright / leaning / horizontal

        # --- bounded histories (all aligned with `timestamps`) ---
        self.timestamps = deque(maxlen=n)
        self.center_history = deque(maxlen=n)          # position history
        self.foot_history = deque(maxlen=n)
        self.velocity_history = deque(maxlen=n)
        self.accel_history = deque(maxlen=n)           # acceleration magnitude
        self.height_history = deque(maxlen=n)
        self.torso_angle_history = deque(maxlen=n)
        self.horizontal_history = deque(maxlen=n)
        self.pose_extent_history = deque(maxlen=n)
        self.posture_history = deque(maxlen=n)         # upright / leaning / horizontal
        self.relative_pose_history = deque(maxlen=n)
        self.shape_history = deque(maxlen=n)
        # not aligned with timestamps:
        self.behavior_history = deque(maxlen=n)
        self.sit_score_history = deque(maxlen=n)       # (time, score)

        # --- behavior ---
        self.state = BehaviorState.NORMAL
        self.state_since = timestamp
        self.behavior = "ANALYZING"
        self.behavior_confidence = 0.0
        self.suspicion_category = None
        self.suspicion_reasons = []
        self.event_start_time = None
        self.last_alert_time = None
        self.had_fall = False
        self.fall_memory_score = 0.0
        self.fall_memory_time = -1e9
        self.active_events = {}                        # event_type -> event dict
        self._timers = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def timer(self, name: str) -> ConditionTimer:
        if name not in self._timers:
            self._timers[name] = ConditionTimer(cfg.TIMER_GRACE_SEC)
        return self._timers[name]

    @property
    def warmed_up(self) -> bool:
        return len(self.timestamps) >= 2 and (self.timestamps[-1] - self.timestamps[0]) >= cfg.WARMUP_SEC

    def record_behavior(self, label: str) -> None:
        self.behavior_history.append(label)

    def index_seconds_ago(self, seconds: float, now=None):
        """Index into the histories of the newest sample that is at least `seconds` old
        (or the oldest sample if history is shorter). None if there is no history."""
        if not self.timestamps:
            return None
        if now is None:
            now = self.timestamps[-1]
        target = now - seconds
        for i in range(len(self.timestamps) - 1, -1, -1):
            if self.timestamps[i] <= target:
                return i
        return 0

    def first_index_within(self, seconds: float) -> int:
        """Index of the first sample inside the last `seconds` seconds."""
        if not self.timestamps:
            return 0
        cutoff = self.timestamps[-1] - seconds
        for i, ts in enumerate(self.timestamps):
            if ts >= cutoff:
                return i
        return len(self.timestamps)

    def window(self, history, seconds: float) -> list:
        return list(history)[self.first_index_within(seconds):]

    def split_history(self, history, seconds: float):
        """(older_items, recent_items) split at `seconds` before now."""
        items = list(history)
        split = self.first_index_within(seconds)
        return items[:split], items[split:]

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(self, bbox, keypoints_xy, keypoints_conf, frame_index, timestamp, escalator_direction):
        x1, y1, x2, y2 = (float(v) for v in bbox)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        self.previous_center = self.center.copy() if self.timestamps else None
        self.bbox = (x1, y1, x2, y2)
        self.bbox_width, self.bbox_height = width, height
        self.aspect_ratio = width / height
        self.center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        self.foot_point = np.array([(x1 + x2) / 2.0, y2])

        raw_scale = max(width, height)
        if not self.timestamps:
            self.body_scale = raw_scale
        else:
            a = cfg.BODY_SCALE_SMOOTHING
            self.body_scale = (1 - a) * self.body_scale + a * raw_scale

        self.last_seen_time = timestamp
        self.last_frame_index = frame_index

        self._update_pose(keypoints_xy, keypoints_conf)
        self._update_horizontal_score()
        self._update_kinematics(timestamp, escalator_direction)   # uses OLD history
        self._update_internal_motion(timestamp)                   # uses OLD history
        self._append_history(timestamp)

    # ------------------------------------------------------------------
    def _reset_pose_features(self):
        self.has_pose = False
        self.keypoints_xy = None
        self.keypoints_conf = None
        self.torso_angle = math.nan
        self.torso_angle_smooth = math.nan
        self.knee_angles = (math.nan, math.nan)
        self.mean_knee_angle = math.nan
        self.hip_ratio = math.nan
        self.leg_extent_ratio = math.nan
        self.extent_aspect = math.nan
        self.pose_extent = math.nan

    def _update_pose(self, xy, conf):
        self._reset_pose_features()
        if xy is None or conf is None:
            return
        xy = np.asarray(xy, dtype=np.float64)
        conf = np.asarray(conf, dtype=np.float64)
        if xy.shape != (17, 2) or conf.shape != (17,):
            return
        valid = (conf >= cfg.KEYPOINT_CONF_THRESHOLD) & np.all(np.isfinite(xy), axis=1)
        if int(valid.sum()) < cfg.MIN_VALID_KEYPOINTS:
            return                                  # too little information: ignore this pose
        kp = xy.copy()
        kp[~valid] = np.nan                         # missing keypoints -> NaN
        self.keypoints_xy = kp
        self.keypoints_conf = conf
        self.has_pose = True

        angle, torso_length = torso_geometry(kp)
        self.torso_angle = angle
        if is_finite(angle):
            prev = self.torso_angle_history[-1] if self.torso_angle_history else math.nan
            a = cfg.TORSO_SMOOTHING
            self.torso_angle_smooth = angle if not is_finite(prev) else (1 - a) * prev + a * angle

        left = angle_at(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        right = angle_at(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
        self.knee_angles = (left, right)
        finite_knees = [k for k in (left, right) if is_finite(k)]
        if finite_knees:
            self.mean_knee_angle = float(np.mean(finite_knees))

        hips = midpoint(kp, L_HIP, R_HIP)
        ankles = midpoint(kp, L_ANKLE, R_ANKLE)
        if hips is not None:
            self.hip_ratio = (hips[1] - self.bbox[1]) / self.bbox_height
        if hips is not None and ankles is not None and is_finite(torso_length) and torso_length > 1e-6:
            self.leg_extent_ratio = abs(ankles[1] - hips[1]) / torso_length

        points = kp[valid]
        x_extent = float(points[:, 0].max() - points[:, 0].min())
        y_extent = float(points[:, 1].max() - points[:, 1].min())
        self.extent_aspect = x_extent / max(y_extent, 1.0)
        self.pose_extent = y_extent / max(self.body_scale, 1.0)

    def _update_horizontal_score(self):
        """
        Fuse several weak signals into one 0..1 'how horizontal is this body' score:
          torso angle (0.4), bbox aspect ratio (0.3), keypoint x/y extent (0.3).
        Missing signals are skipped and the weights renormalised, so a missing pose
        degrades gracefully to the bounding box.
        """
        signals = []
        if is_finite(self.torso_angle_smooth):
            signals.append((0.4, ramp(self.torso_angle_smooth, *cfg.HORIZ_TORSO_RANGE)))
        signals.append((0.3, ramp(self.aspect_ratio, *cfg.HORIZ_ASPECT_RANGE)))
        if is_finite(self.extent_aspect):
            signals.append((0.3, ramp(self.extent_aspect, *cfg.HORIZ_EXTENT_RANGE)))
        total_weight = sum(w for w, _ in signals)
        raw = sum(w * v for w, v in signals) / total_weight
        if self.timestamps:
            a = cfg.HORIZONTAL_SMOOTHING
            self.horizontal_score = (1 - a) * self.horizontal_score + a * raw
        else:
            self.horizontal_score = raw
        if self.horizontal_score >= cfg.HORIZONTAL_SCORE_MIN:
            self.body_orientation = "horizontal"
        elif self.horizontal_score <= cfg.UPRIGHT_SCORE_MAX:
            self.body_orientation = "upright"
        else:
            self.body_orientation = "leaning"

    def _update_kinematics(self, now, escalator_direction):
        scale = max(self.body_scale, 1.0)

        # Velocity = displacement over a short lag (acts as noise smoothing), in BH/s.
        idx = self.index_seconds_ago(cfg.VELOCITY_LAG_SEC, now)
        if idx is None:
            self.velocity = np.zeros(2)
        else:
            dt = now - self.timestamps[idx]
            if dt > 1e-3:
                self.velocity = (self.center - np.asarray(self.center_history[idx])) / dt / scale

        # Acceleration = change of velocity over a lag (differentiating twice amplifies noise,
        # so a longer lag than one frame is essential).
        idx = self.index_seconds_ago(cfg.ACCEL_LAG_SEC, now)
        if idx is None:
            self.acceleration = np.zeros(2)
        else:
            dt = now - self.timestamps[idx]
            if dt > 1e-3:
                self.acceleration = (self.velocity - np.asarray(self.velocity_history[idx])) / dt

        # Escalator-relative motion over a slightly longer window.
        window_velocity = np.zeros(2)
        idx = self.index_seconds_ago(cfg.DIRECTION_WINDOW_SEC, now)
        if idx is not None:
            dt = now - self.timestamps[idx]
            if dt > 1e-3:
                window_velocity = (self.center - np.asarray(self.center_history[idx])) / dt / scale

        direction = np.asarray(escalator_direction, dtype=np.float64)   # unit vector
        self.speed = float(np.linalg.norm(window_velocity))
        self.along_speed = float(np.dot(window_velocity, direction))
        # Component perpendicular to the escalator: v - (v.d) d. Being carried by the
        # escalator contributes nothing here, so it measures "moving on its own".
        off_axis = window_velocity - self.along_speed * direction
        self.off_axis_speed = float(np.linalg.norm(off_axis))
        self.direction_cosine = cosine_similarity(window_velocity, direction)

        if self.speed < cfg.MOVING_MIN_SPEED:
            self.movement_direction = "stationary"
        elif self.direction_cosine >= cfg.DIRECTION_COS_WITH:
            self.movement_direction = "with"
        elif self.direction_cosine <= -cfg.DIRECTION_COS_WITH:
            self.movement_direction = "against"
        else:
            self.movement_direction = "lateral"

    def _update_internal_motion(self, now):
        """
        Motion of the body parts RELATIVE to the body centre, in BH/s. A person lying
        still on a moving escalator has ~0 internal motion even though they move in the
        image, which is exactly what 'still' should mean here.
        """
        idx = self.index_seconds_ago(cfg.INTERNAL_MOTION_LAG_SEC, now)
        if idx is None:
            return
        dt = now - self.timestamps[idx]
        if dt <= 1e-3:
            return
        scale = max(self.body_scale, 1.0)
        value = None
        previous_relative = self.relative_pose_history[idx]
        if self.has_pose and previous_relative is not None:
            diff = (self.keypoints_xy - self.center) - previous_relative
            ok = np.all(np.isfinite(diff), axis=1)
            if int(ok.sum()) >= 4:
                value = float(np.mean(np.linalg.norm(diff[ok], axis=1))) / scale / dt
        if value is None:   # fallback: change of the box shape
            prev_w, prev_h = self.shape_history[idx]
            value = (abs(self.bbox_width - prev_w) + abs(self.bbox_height - prev_h)) / 2.0 / scale / dt
        self.internal_motion = 0.6 * self.internal_motion + 0.4 * value

    def _append_history(self, now):
        self.timestamps.append(now)
        self.center_history.append((float(self.center[0]), float(self.center[1])))
        self.foot_history.append((float(self.foot_point[0]), float(self.foot_point[1])))
        self.velocity_history.append((float(self.velocity[0]), float(self.velocity[1])))
        self.accel_history.append(float(np.linalg.norm(self.acceleration)))
        self.height_history.append(self.bbox_height)
        self.torso_angle_history.append(self.torso_angle_smooth)
        self.horizontal_history.append(self.horizontal_score)
        self.pose_extent_history.append(self.pose_extent)
        self.posture_history.append(self.body_orientation)
        self.relative_pose_history.append(
            (self.keypoints_xy - self.center) if self.has_pose else None)
        self.shape_history.append((self.bbox_width, self.bbox_height))

