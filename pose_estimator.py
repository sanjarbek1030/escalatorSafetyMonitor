"""
pose_estimator.py - YOLO-Pose keypoints, matched to tracked people.

The pose model is a separate network with its own boxes (no track IDs). Each pose is
matched to a tracked person by bounding-box IoU (greedy, one-to-one).
Pose is run on the ROI crop only, which saves compute and gives small people more pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as cfg
from utils import iou


@dataclass
class PoseResult:
    xy: np.ndarray        # (17, 2) full-frame pixel coordinates
    conf: np.ndarray      # (17,)   0 for missing keypoints


class PoseEstimator:
    def __init__(self, model_path: str, device: str, use_half: bool):
        from ultralytics import YOLO
        self.device = device
        self.use_half = use_half
        self.model = YOLO(model_path)

    def estimate(self, frame, detections, crop_rect=None) -> dict:
        """Returns {track_id: PoseResult}. Never raises for empty or missing poses."""
        if not detections:
            return {}
        x_offset, y_offset = 0, 0
        image = frame
        if crop_rect is not None:
            x0, y0, x1, y1 = crop_rect
            if x1 - x0 < 32 or y1 - y0 < 32:
                return {}
            image = frame[y0:y1, x0:x1]          # a view, no copy
            x_offset, y_offset = x0, y0

        results = self.model.predict(
            image, conf=cfg.POSE_CONFIDENCE, imgsz=cfg.POSE_IMGSZ,
            device=self.device, half=self.use_half, verbose=False)
        if not results:
            return {}
        result = results[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return {}

        pose_boxes = result.boxes.xyxy.cpu().numpy()
        keypoints_xy = result.keypoints.xy.cpu().numpy()                 # (N, 17, 2)
        if result.keypoints.conf is not None:
            keypoints_conf = result.keypoints.conf.cpu().numpy()         # (N, 17)
        else:
            keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)
        if keypoints_xy.ndim != 3 or keypoints_xy.shape[1] != 17:
            return {}

        # Undetected keypoints come back as (0, 0): mark them as confidence 0 BEFORE
        # adding the crop offset, otherwise they would look like real points.
        missing = (keypoints_xy[..., 0] <= 0) & (keypoints_xy[..., 1] <= 0)
        keypoints_conf = np.where(missing, 0.0, keypoints_conf)
        pose_boxes = pose_boxes + np.array([x_offset, y_offset, x_offset, y_offset])
        keypoints_xy = keypoints_xy + np.array([x_offset, y_offset])

        candidates = []
        for det_index, det in enumerate(detections):
            for pose_index in range(len(pose_boxes)):
                score = iou(det.bbox, pose_boxes[pose_index])
                if score >= cfg.POSE_MATCH_IOU:
                    candidates.append((score, det_index, pose_index))
        candidates.sort(reverse=True)

        used_dets, used_poses, pose_map = set(), set(), {}
        for _, det_index, pose_index in candidates:
            if det_index in used_dets or pose_index in used_poses:
                continue
            used_dets.add(det_index)
            used_poses.add(pose_index)
            pose_map[detections[det_index].track_id] = PoseResult(
                xy=keypoints_xy[pose_index].astype(np.float64),
                conf=keypoints_conf[pose_index].astype(np.float64))
        return pose_map

