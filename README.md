# Escalator Safety Monitor

An **automated visual warning system** that analyzes escalator CCTV video and flags
potentially unusual events (sitting, falling, person down, reverse movement, possible
crowd gathering) **for human review**. It is a research / portfolio project, not a certified
safety system. False positives and false negatives are possible.

## Overview
The pipeline combines two deep-learning models (YOLO person detection, YOLO pose
estimation), a classical tracker (ByteTrack) and a **hand-designed heuristic temporal
analysis layer**. There is **no learned behavior classifier** in this version.

| Component | Type |
|---|---|
| Person detection (YOLO11n) | Deep learning |
| Pose estimation (YOLO11n-pose) | Deep learning |
| Tracking (ByteTrack / BoT-SORT) | Classical association algorithm |
| Fall, sitting, unusual-behavior scoring, state machine | Manually designed rules |
| Confidence values | Composite heuristic scores, **not** calibrated probabilities |

## Features
- Escalator ROI polygon and direction, both configurable, plus a click-to-configure tool
- Persistent track IDs, with analysis restricted to people on the escalator
- Skeleton overlay, robust to missing keypoints
- Speed and acceleration measured in body-lengths so they do not depend on resolution
- Escalator-relative motion: being carried by the escalator is not counted as movement
- Multi-signal fall scoring over a rolling time window with post-fall confirmation
- Sitting detection with temporal persistence, and person-down detection
- Unusual-behavior score (reverse movement, crouching, crawling, sudden acceleration, ...)
- Per-person state machine with hysteresis: NORMAL, SUSPICIOUS, SITTING, FALLING, PERSON_DOWN, RECOVERING
- Basic crowd-gathering analysis around a fallen person
- Alert cooldown, alert banner, JSON event log, one screenshot per event
- Automatic GPU use with CPU fallback

## Architecture
```
frame -> PersonDetector (YOLO + ByteTrack) -> CrowdAnalyzer (all tracks)
      -> ROI filter -> PoseEstimator (YOLO-Pose on ROI crop) -> PersonState
      -> BehaviorAnalyzer (Fall / Sitting / Unusual + StateMachine)
      -> AlertManager + EventLogger -> Visualizer -> output video
```

## Project structure
```
main.py  config.py  utils.py  detector.py  tracker.py  pose_estimator.py
person_state.py  escalator.py  behavior_analyzer.py  fall_detector.py
sitting_detector.py  crowd_analyzer.py  alert_manager.py  event_logger.py
visualization.py  roi_picker.py  requirements.txt
input/escalator_input.mp4        (you provide)
output/escalator_result.mp4, output/events.json, output/events/*.jpg
```

## Installation (Windows / PyCharm)
```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy
```
Requirements: Python 3.9+, `ultralytics`, `opencv-python`, `numpy` (PyTorch is installed
with Ultralytics).

Optional NVIDIA GPU (choose the command for your CUDA version at pytorch.org):
```bat
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## How to run
1. Put your video at `input/escalator_input.mp4`.
2. `python roi_picker.py`, click the escalator polygon and direction, and paste the printed lines into `config.py`.
3. `python main.py` (add `--no-preview` to disable the window).

Results: `output/escalator_result.mp4`, `output/events.json`, `output/events/`.

## Configuration
Everything is in `config.py`: paths, models, `CONFIDENCE_THRESHOLD`, `ESCALATOR_POLYGON`,
`ESCALATOR_DIRECTION`, `SITTING_DURATION`, `FALL_WEIGHTS`, `FALL_RANGES`,
`FALL_SCORE_THRESHOLD`, `PERSON_DOWN_DURATION`, `ALERT_COOLDOWN`, `HISTORY_LENGTH`, and more.
Thresholds are starting points and must be tuned on your own footage.

## How it works: detection pipeline
1. YOLO detects people (COCO class 0); ByteTrack assigns persistent IDs.
2. People whose feet or body centre are inside the ROI are kept.
3. YOLO-Pose runs on the ROI crop; poses are matched to track IDs by IoU.
4. `PersonState` computes geometry, escalator-relative kinematics and pose features and stores bounded histories.
5. Detectors score the person; the state machine confirms transitions over time.
6. Confirmed events are logged, alerted (with cooldown) and drawn.

## Fall detection methodology
A weighted sum of nine pieces of evidence: velocity deviation, downward displacement,
acceleration, torso rotation, bbox height drop, pose collapse, horizontal configuration,
upright-to-horizontal transition and post-fall stillness. Dynamic evidence is peak-held for
5 s. `FALLING` needs score >= threshold, a fairly horizontal body and 0.3 s of confirmation.
A person simply lying down without a preceding fall dynamic does not reach the threshold.

## Sitting detection methodology
Knee angle, hip position, folded legs, compact box shape and upright torso are fused into a
score that must persist for `SITTING_DURATION`. Pose is required; a box alone never confirms sitting.

## Tracking methodology
ByteTrack (Kalman filter + IoU association including low-confidence boxes) via
`model.track(persist=True)`. `TrackManager` owns one `PersonState` per ID and expires it after a timeout.

## Example output
Annotated video with ROI overlay, skeletons, labels such as `PERSON #17 / FALLING 91%`, HUD,
and a red alert banner; `events.json` with one entry per event; one screenshot per event.

## Limitations
- Not validated on any dataset; all thresholds are hand-set heuristics
- Depends on camera angle, lighting, occlusion and pose quality
- ByteTrack has no appearance re-identification, so IDs may switch after long occlusions
- Overhead cameras hide legs, which weakens sitting and pose-based cues
- Children, wheelchairs, luggage or people bending can cause false alarms
- Crowd analysis uses image-plane distances and ignores perspective
- Not a guaranteed safety system: every alert needs human review

## Future improvements
Camera calibration (ground-plane distances), appearance re-ID, more test footage with labeled
events, automatic ROI/direction estimation, evaluation scripts (precision/recall per event),
a web dashboard, RTSP live streams.

## Machine-learning upgrade path
Pose sequence (30-60 frames) -> normalization -> GRU / LSTM / Transformer -> behavior class
(normal, sitting, falling, lying, crouching, reverse_movement). This would be a **separate
learned component** and must be split by recording/camera to avoid leakage.
See the project documentation for details.

## Disclaimer
Research and portfolio use only. Not a certified safety product.