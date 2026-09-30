"""
config.py - every tunable parameter of the Escalator Safety Monitor.

Units used throughout:
  *_SEC / *_DURATION : seconds of VIDEO time (frame_index / fps), not wall-clock time
  speeds             : body-lengths per second (BH/s). 1.0 means "moves one body length per second"
  accelerations      : BH/s^2
  angles             : degrees
  *_RANGE = (low, high): a soft ramp. Evidence is 0 at/below 'low', 1 at/above 'high'.

IMPORTANT: all confidence values are COMPOSITE BEHAVIOR SCORES built from hand-designed
rules. They are NOT statistically calibrated probabilities.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (relative to this file, so PyCharm's working directory does not matter)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_VIDEO = PROJECT_ROOT / "input" / "escalator_input.mp4"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_VIDEO = OUTPUT_DIR / "escalator_result.mp4"
EVENTS_JSON = OUTPUT_DIR / "events.json"
EVENT_SCREENSHOT_DIR = OUTPUT_DIR / "events"
CLEAR_OLD_SCREENSHOTS = True      # delete old *.jpg in the screenshot folder at start
OUTPUT_FOURCC = "mp4v"            # most compatible with pip opencv; try "avc1" for H.264
SHOW_PREVIEW = True               # live window while processing (press q to stop)
FALLBACK_FPS = 25.0               # used only if the video has no valid FPS metadata

# ----------------------------------------------------------------------------
# Models and device
# ----------------------------------------------------------------------------
MODEL_PATH = "yolo11n.pt"             # YOLO detection (auto-downloaded on first run)
POSE_MODEL_PATH = "yolo11n-pose.pt"   # YOLO pose (auto-downloaded on first run)
FORCE_CPU = False
USE_HALF_PRECISION = True             # FP16, only applied when CUDA is used

# ----------------------------------------------------------------------------
# Detection and tracking
# ----------------------------------------------------------------------------
PERSON_CLASS_ID = 0                   # COCO class 0 = person
CONFIDENCE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.50
DETECTION_IMGSZ = 640                 # raise to 960/1280 for small people (slower)
TRACKER_CONFIG = "bytetrack.yaml"     # or "botsort.yaml"
LOST_TRACK_TIMEOUT_SEC = 2.0          # forget a track this long after it was last seen
DANGER_TRACK_TIMEOUT_SEC = 10.0       # keep FALLING / PERSON_DOWN tracks longer
WARMUP_SEC = 0.6                      # a new track is only "ANALYZING" for this long

# ----------------------------------------------------------------------------
# Pose
# ----------------------------------------------------------------------------
POSE_CONFIDENCE = 0.25
POSE_IMGSZ = 640
POSE_MATCH_IOU = 0.30                 # min IoU to match a pose box to a tracked box
KEYPOINT_CONF_THRESHOLD = 0.35        # keypoints below this are treated as missing
MIN_VALID_KEYPOINTS = 6               # fewer than this -> pose ignored for that frame
POSE_CROP_MARGIN_FRACTION = 0.05      # pose runs only on the ROI bounding box + margin
DRAW_SKELETON = True

# ----------------------------------------------------------------------------
# Escalator ROI and direction
# ----------------------------------------------------------------------------
# Polygon corners in pixels of a REFERENCE frame size. If your video has a different
# size the polygon is scaled automatically. Set REFERENCE_FRAME_SIZE = None if the
# coordinates are already in the real video's pixel coordinates.
# Use  python roi_picker.py  to click the corners on your own video.
REFERENCE_FRAME_SIZE = (1280, 720)
ESCALATOR_POLYGON = [
    (490, 130),    # top-left | original 420, 120
    (830, 130),    # top-right | original 860, 120
    (1300, 700),   # bottom-right | original 1000, 700
    (80, 700),    # bottom-left | original 280, 700
]
# Direction people are carried by the escalator, in IMAGE coordinates (x right, y DOWN).
# (0, -1) = people travel toward the top of the image. (0, 1) = toward the bottom.
ESCALATOR_DIRECTION = (0.0, -1.0)
ROI_COLOR = (255, 160, 0)             # BGR
ROI_ALPHA = 0.22

# ----------------------------------------------------------------------------
# Person history / smoothing
# ----------------------------------------------------------------------------
HISTORY_SECONDS = 3.0                 # history length in seconds ...
HISTORY_MIN_LENGTH = 30               # ... clamped to [30, 90] frames
HISTORY_LENGTH = 90
BODY_SCALE_SMOOTHING = 0.2
TORSO_SMOOTHING = 0.5
HORIZONTAL_SMOOTHING = 0.5
TIMER_GRACE_SEC = 0.3                 # a condition may drop out this long without resetting its timer
MIN_TORSO_PIXELS = 6.0

# ----------------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------------
VELOCITY_LAG_SEC = 0.15               # velocity = displacement over this lag (noise suppression)
ACCEL_LAG_SEC = 0.25
DIRECTION_WINDOW_SEC = 0.5            # window used for escalator-direction analysis
MOVING_MIN_SPEED = 0.12               # below this -> "stationary"
DIRECTION_COS_WITH = 0.40             # cosine > +0.4 "with", < -0.4 "against"
INTERNAL_MOTION_LAG_SEC = 0.3

# ----------------------------------------------------------------------------
# Posture (horizontal-ness)
# ----------------------------------------------------------------------------
HORIZ_TORSO_RANGE = (30.0, 60.0)      # torso angle from vertical (deg)
HORIZ_ASPECT_RANGE = (0.8, 1.4)       # bbox width / height
HORIZ_EXTENT_RANGE = (0.8, 1.5)       # keypoint x-extent / y-extent
UPRIGHT_SCORE_MAX = 0.25              # horizontal_score <= this counts as upright
HORIZONTAL_SCORE_MIN = 0.60           # horizontal_score >= this counts as horizontal

# ----------------------------------------------------------------------------
# Fall detection (all weights and ranges are tunable)
# ----------------------------------------------------------------------------
FALL_WINDOW_SEC = 1.0                 # rolling window for dynamic evidence
FALL_LOOKBACK_SEC = 2.0               # look-back for height drop / upright-before-fall
FALL_MEMORY_SEC = 5.0                 # dynamic evidence is peak-held this long
FALL_MEMORY_MIN_DYNAMIC = 3.0         # min remembered dynamic score to allow 'post_fall' points
POST_FALL_STILL_SEC = 0.5

FALL_WEIGHTS = {
    "velocity": 1.5,
    "acceleration": 1.5,
    "displacement": 1.5,
    "rotation": 2.0,
    "height_drop": 1.5,
    "collapse": 1.0,
    "horizontal": 2.0,
    "transition": 2.0,
    "post_fall": 3.0,
}
FALL_RANGES = {
    "velocity": (0.6, 1.6),           # BH/s deviation from baseline velocity
    "acceleration": (2.0, 6.0),       # BH/s^2
    "displacement": (0.15, 0.50),     # downward body lengths beyond baseline
    "rotation": (20.0, 50.0),         # degrees of torso rotation in the window
    "height_drop": (0.15, 0.45),      # fractional decrease of bbox height
    "collapse": (0.25, 0.60),         # fractional decrease of keypoint vertical extent
    "horizontal": (0.35, 0.75),       # horizontal_score ramp
}
FALL_SCORE_THRESHOLD = 6.0            # score needed for FALLING (max possible = sum of weights = 16)
FALL_SCORE_SATURATION = 11.0          # score that maps to confidence 1.0
FALL_MIN_HORIZONTAL_SCORE = 0.45      # FALLING also needs a fairly horizontal body
FALL_CONFIRM_SEC = 0.30               # score must hold this long

# ----------------------------------------------------------------------------
# Person down / recovery
# ----------------------------------------------------------------------------
PERSON_DOWN_DURATION = 1.5            # horizontal + still, after a detected fall
PERSON_DOWN_NO_FALL_DURATION = 3.0    # horizontal + still, but the fall itself was not observed
STILL_INTERNAL_MOTION_MAX = 0.12      # BH/s of body-internal (pose) motion
STILL_OFFAXIS_SPEED_MAX = 0.25        # BH/s of motion across the escalator direction
FALLING_MAX_SEC = 6.0
RECOVERED_HORIZONTAL_SCORE = 0.35     # below this the body counts as "back up"
RECOVER_CONFIRM_SEC = 1.0
RECOVERY_DURATION_SEC = 2.0

# ----------------------------------------------------------------------------
# Sitting detection
# ----------------------------------------------------------------------------
SITTING_DURATION = 1.5                # posture must persist this long
SIT_SCORE_THRESHOLD = 0.60
SIT_SCORE_EXIT_THRESHOLD = 0.45
SIT_EXIT_SEC = 0.7
SIT_MAX_HORIZONTAL_SCORE = 0.50
SIT_MIN_AVAILABLE_WEIGHT = 0.50       # sitting needs pose; bbox alone is never enough
SIT_WEIGHTS = {"knee": 0.30, "hip_low": 0.20, "legs_folded": 0.20,
               "compact_box": 0.15, "torso_upright": 0.15}
SIT_KNEE_TRAPEZOID = (40.0, 70.0, 125.0, 155.0)   # zero, full, full, zero (deg)
SIT_HIP_RATIO_RANGE = (0.50, 0.68)                # hip position inside bbox (0 top .. 1 bottom)
SIT_LEG_EXTENT_RANGE = (1.0, 1.9)                 # |ankle_y - hip_y| / torso length (low = folded)
SIT_ASPECT_TRAPEZOID = (0.35, 0.55, 1.10, 1.50)   # bbox width / height
SIT_TORSO_RANGE = (25.0, 50.0)

# ----------------------------------------------------------------------------
# Unusual behavior (SUSPICIOUS)
# ----------------------------------------------------------------------------
UNUSUAL_WEIGHTS = {
    "reverse": 0.40,
    "sudden_acceleration": 0.25,
    "abnormal_torso": 0.25,
    "crouching": 0.30,
    "crawling": 0.35,
    "long_stationary": 0.20,
    "approaching_person": 0.20,
    "rapid_rotation": 0.25,
}
SUSPICIOUS_SCORE_ENTER = 0.35
SUSPICIOUS_SCORE_EXIT = 0.20
SUSPICIOUS_ENTER_SEC = 0.6
SUSPICIOUS_EXIT_SEC = 1.0
UNUSUAL_EVENT_MIN_SEC = 3.0           # SUSPICIOUS must last this long to be logged as an event
REVERSE_MIN_SPEED = 0.30
REVERSE_MIN_SEC = 1.0
SUSPICIOUS_ACCEL_RANGE = (2.5, 6.0)
ABNORMAL_TORSO_RANGE = (25.0, 50.0)
CROUCH_KNEE_MAX = 100.0
CROUCH_HEIGHT_RATIO = 0.75
CROUCH_MIN_SEC = 2.0
CRAWL_HORIZONTAL_MIN = 0.50
CRAWL_INTERNAL_MOTION_MIN = 0.30
CRAWL_MIN_SEC = 1.0
LONG_STATIONARY_SEC = 10.0
APPROACH_SPEED_RANGE = (0.6, 1.8)
ROTATION_RATE_WINDOW_SEC = 0.4
ROTATION_RATE_RANGE = (80.0, 200.0)   # deg/s

# ----------------------------------------------------------------------------
# Crowd interaction
# ----------------------------------------------------------------------------
CROWD_LOOKBACK_SEC = 1.0
CROWD_RADIUS_BODY_SCALES = 2.0        # "around the fallen person" = within 2 body lengths
CROWD_ARRIVAL_FACTOR = 1.3            # was farther than radius*1.3 one lookback ago
CROWD_MIN_NEAR = 3
CROWD_MIN_ARRIVING = 2
CROWD_CONFIRM_SEC = 1.0
CROWD_GRACE_SEC = 1.0
CROWD_TRACK_TIMEOUT_SEC = 1.5

# ----------------------------------------------------------------------------
# Alerts and events
# ----------------------------------------------------------------------------
ALERT_COOLDOWN = 10.0                 # seconds, per (event type, track id)
BANNER_DURATION_SEC = 4.0
BANNER_EVENT_TYPES = ("fall", "person_down", "crowd_incident")
EVENT_MERGE_SECONDS = 5.0             # same track+type reappearing within this re-opens the old event
