"""escalator.py - the escalator Region Of Interest (ROI) and expected movement direction."""
from __future__ import annotations

import cv2
import numpy as np


class EscalatorRegion:
    def __init__(self, polygon, direction, frame_width, frame_height, reference_size=None):
        if polygon is None or len(polygon) < 3:
            raise ValueError("ESCALATOR_POLYGON needs at least 3 points.")
        points = np.array(polygon, dtype=np.float64)
        if reference_size:                      # scale from the reference size to the real video size
            points[:, 0] *= frame_width / float(reference_size[0])
            points[:, 1] *= frame_height / float(reference_size[1])
        points[:, 0] = np.clip(points[:, 0], 0, frame_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, frame_height - 1)
        self.polygon_int = np.round(points).astype(np.int32)          # shape (N, 2)
        self._contour = self.polygon_int.reshape(-1, 1, 2)            # shape OpenCV expects
        self.frame_width = frame_width
        self.frame_height = frame_height

        direction = np.array(direction, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            raise ValueError("ESCALATOR_DIRECTION must not be (0, 0).")
        self.direction = direction / length                            # unit vector

    def contains_point(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self._contour, (float(x), float(y)), False) >= 0

    def contains_person(self, bbox) -> bool:
        """A person is 'on the escalator' if their feet OR body centre are inside the polygon."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        return (self.contains_point(center_x, y2)
                or self.contains_point(center_x, (y1 + y2) / 2.0))

    def bounding_rect(self, margin_fraction: float):
        """(x0, y0, x1, y1) of the polygon plus a margin, clipped to the frame."""
        x, y, w, h = cv2.boundingRect(self.polygon_int)
        margin_x = int(margin_fraction * self.frame_width)
        margin_y = int(margin_fraction * self.frame_height)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(self.frame_width, x + w + margin_x)
        y1 = min(self.frame_height, y + h + margin_y)
        return x0, y0, x1, y1

    def centroid(self):
        return tuple(np.mean(self.polygon_int, axis=0).astype(int))
