"""
detector.py - YOLO person detection + ByteTrack/BoT-SORT tracking.

Ultralytics runs the detector and the tracker in a single call (model.track). The YOLO
network is the learned part; ByteTrack is a classical association algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass

import config as cfg


def select_device(force_cpu: bool = False):
    """Returns (device_string, is_cuda) and clearly prints which one is used."""
    import torch
    if not force_cpu and torch.cuda.is_available():
        print(f"[DEVICE] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda:0", True
    if force_cpu:
        print("[DEVICE] Using CPU (FORCE_CPU=True in config.py)")
    else:
        print("[DEVICE] CUDA not available -> using CPU")
        if torch.version.cuda is None:
            print("         (installed PyTorch is a CPU-only build; see README for the CUDA install)")
    return "cpu", False


@dataclass
class TrackedDetection:
    track_id: int
    bbox: tuple            # (x1, y1, x2, y2) in pixels
    confidence: float


class PersonDetector:
    def __init__(self, model_path: str, device: str, use_half: bool):
        from ultralytics import YOLO
        self.device = device
        self.use_half = use_half
        self.model = YOLO(model_path)     # downloads the weights automatically if missing

    def detect_and_track(self, frame) -> list:
        """Detect people and return detections that have a persistent track ID."""
        results = self.model.track(
            frame,
            persist=True,                         # keep track IDs between frames
            tracker=cfg.TRACKER_CONFIG,
            classes=[cfg.PERSON_CLASS_ID],        # people only
            conf=cfg.CONFIDENCE_THRESHOLD,
            iou=cfg.NMS_IOU_THRESHOLD,
            imgsz=cfg.DETECTION_IMGSZ,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        # boxes.id is None when nothing was tracked in this frame
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        track_ids = boxes.id.int().cpu().numpy()
        detections = []
        for box, conf, track_id in zip(xyxy, confidences, track_ids):
            detections.append(TrackedDetection(
                track_id=int(track_id),
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(conf)))
        return detections
