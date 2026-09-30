"""visualization.py - everything that is DRAWN on the output video."""
from __future__ import annotations

import cv2
import numpy as np

import config as cfg
from person_state import DANGER_STATES, SKELETON_EDGES, BehaviorState
from utils import clip

STATE_COLORS = {                       # BGR
    BehaviorState.NORMAL: (80, 200, 80),
    BehaviorState.SUSPICIOUS: (0, 200, 255),
    BehaviorState.SITTING: (255, 170, 0),
    BehaviorState.FALLING: (0, 60, 255),
    BehaviorState.PERSON_DOWN: (0, 0, 220),
    BehaviorState.RECOVERING: (200, 200, 0),
}


class Visualizer:
    def __init__(self, escalator, frame_width: int, frame_height: int):
        self.escalator = escalator
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ui = max(0.6, frame_height / 720.0)          # scales text/lines with resolution
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.hud_right = 0

        x, y, w, h = cv2.boundingRect(escalator.polygon_int)
        self.roi_x0, self.roi_y0 = max(0, x), max(0, y)
        self.roi_x1, self.roi_y1 = min(frame_width, x + w), min(frame_height, y + h)
        self.roi_shifted = (escalator.polygon_int - np.array([self.roi_x0, self.roi_y0])).astype(np.int32)

    # ------------------------------------------------------------------ helpers
    def _blend_rect(self, frame, x0, y0, x1, y1, color, alpha):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        region = frame[y0:y1, x0:x1]
        overlay = np.full_like(region, color)
        frame[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, region, 1.0 - alpha, 0)

    @staticmethod
    def _contrast_color(bg):
        luminance = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
        return (0, 0, 0) if luminance > 140 else (255, 255, 255)

    def _text(self, frame, text, origin, scale, color=None, thickness=1, bg=None, pad=4):
        """Text with an optional filled background. `origin` is the bottom-left corner."""
        if color is None:
            color = self._contrast_color(bg) if bg is not None else (255, 255, 255)
        (tw, th), base = cv2.getTextSize(text, self.font, scale, thickness)
        h, w = frame.shape[:2]
        x = int(clip(origin[0], pad, max(pad, w - tw - pad)))
        y = int(clip(origin[1], th + pad, max(th + pad, h - base - pad)))
        if bg is not None:
            cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + base + pad - 1), bg, -1)
        cv2.putText(frame, text, (x, y), self.font, scale, color, thickness, cv2.LINE_AA)

    def _centered_text(self, frame, text, x_left, x_right, baseline, scale, color, thickness):
        available = x_right - x_left - 10
        (tw, _), _ = cv2.getTextSize(text, self.font, scale, thickness)
        while tw > available and scale > 0.35:
            scale -= 0.05
            (tw, _), _ = cv2.getTextSize(text, self.font, scale, thickness)
        x = x_left + max(0, (x_right - x_left - tw) // 2)
        cv2.putText(frame, text, (int(x), int(baseline)), self.font, scale, color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------ ROI
    def draw_roi(self, frame):
        if self.roi_x1 > self.roi_x0 and self.roi_y1 > self.roi_y0:
            region = frame[self.roi_y0:self.roi_y1, self.roi_x0:self.roi_x1]
            overlay = region.copy()
            cv2.fillPoly(overlay, [self.roi_shifted], cfg.ROI_COLOR)
            frame[self.roi_y0:self.roi_y1, self.roi_x0:self.roi_x1] = cv2.addWeighted(
                overlay, cfg.ROI_ALPHA, region, 1.0 - cfg.ROI_ALPHA, 0)
        cv2.polylines(frame, [self.escalator.polygon_int], True, cfg.ROI_COLOR, 2, cv2.LINE_AA)

        top = self.escalator.polygon_int[np.argmin(self.escalator.polygon_int[:, 1])]
        self._text(frame, "ESCALATOR ROI", (int(top[0]), int(top[1]) - 8), 0.6 * self.ui, bg=cfg.ROI_COLOR)

        # arrow showing the configured escalator direction
        cx, cy = self.escalator.centroid()
        length = 0.07 * self.frame_height
        tip = (int(cx + self.escalator.direction[0] * length), int(cy + self.escalator.direction[1] * length))
        cv2.arrowedLine(frame, (int(cx), int(cy)), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)

    # ------------------------------------------------------------------ persons
    def draw_person(self, frame, person):
        color = STATE_COLORS[person.state]
        x1, y1, x2, y2 = (int(v) for v in person.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if person.state in DANGER_STATES else 2)
        if cfg.DRAW_SKELETON and person.has_pose:
            self._draw_skeleton(frame, person.keypoints_xy, color)

        if person.behavior == "ANALYZING":
            line2 = "ANALYZING..."
        else:
            line2 = f"{person.behavior.replace('_', ' ')} {int(round(person.behavior_confidence * 100))}%"
        line1 = f"PERSON #{person.track_id}"
        scale = 0.55 * self.ui
        line_h = int(24 * self.ui)
        if y1 - 2 * line_h - 6 >= 0:
            base2 = y1 - 4
            base1 = base2 - line_h
        else:                                         # not enough room above -> draw below the box
            base1 = y2 + line_h
            base2 = base1 + line_h
        self._text(frame, line1, (x1, base1), scale, bg=(30, 30, 30))
        self._text(frame, line2, (x1, base2), scale, bg=color)
        if person.state == BehaviorState.SUSPICIOUS and person.suspicion_reasons:
            self._text(frame, person.suspicion_reasons[0].replace("_", " "),
                       (x1 + 4, y2 - 6), 0.45 * self.ui, bg=(30, 30, 30), pad=2)

    def _draw_skeleton(self, frame, keypoints, color):
        for a, b in SKELETON_EDGES:
            pa, pb = keypoints[a], keypoints[b]
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2, cv2.LINE_AA)
        for point in keypoints:
            if np.all(np.isfinite(point)):
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_crowd_incident(self, frame, incident):
        center = (int(incident.center[0]), int(incident.center[1]))
        cv2.circle(frame, center, int(incident.radius), (0, 140, 255), 3, cv2.LINE_AA)
        self._text(frame, f"POSSIBLE CROWD INCIDENT {int(round(incident.confidence * 100))}%",
                   (center[0] - int(incident.radius), center[1] - int(incident.radius) - 8),
                   0.6 * self.ui, bg=(0, 140, 255))

    # ------------------------------------------------------------------ HUD
    def draw_hud(self, frame, video_time, processing_fps, people, alerts, status, status_color):
        lines = [
            ("ESCALATOR SAFETY MONITOR", (255, 255, 255)),
            (f"Video time : {video_time}", (255, 255, 255)),
            (f"Processing FPS : {processing_fps:.1f}", (255, 255, 255)),
            (f"People on escalator : {people}", (255, 255, 255)),
            (f"Alerts : {alerts}", (255, 255, 255)),
            (f"STATUS : {status}", status_color),
        ]
        scale = 0.55 * self.ui
        line_h = int(24 * self.ui)
        width = max(cv2.getTextSize(t, self.font, scale, 1)[0][0] for t, _ in lines) + 20
        x0, y0 = 10, 10
        self._blend_rect(frame, x0, y0, x0 + width, y0 + line_h * len(lines) + 10, (0, 0, 0), 0.55)
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (x0 + 10, y0 + line_h * (i + 1)), self.font, scale, color, 1, cv2.LINE_AA)
        self.hud_right = x0 + width

    def draw_footer(self, frame):
        h = frame.shape[0]
        self._text(frame, "Automated visual warning system - all alerts require human review",
                   (10, h - 10), 0.45 * self.ui, bg=(0, 0, 0), pad=3)

    # ------------------------------------------------------------------ banner
    def _warning_icon(self, frame, cx, cy, r):
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (0, 215, 255))
        cv2.polylines(frame, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "!", (cx - int(r * 0.22), cy + int(r * 0.75)), self.font, r / 22.0,
                    (0, 0, 0), 2, cv2.LINE_AA)

    def draw_banner(self, frame, alert, frame_index):
        h, w = frame.shape[:2]
        ui = self.ui
        banner_w = int(min(w * 0.55, 620 * ui))
        banner_h = int(118 * ui)
        x0 = max(self.hud_right + 15, (w - banner_w) // 2)
        if x0 + banner_w > w - 10:
            banner_w = max(50, w - 10 - x0)
        y0 = int(10 * ui)
        pulse = 0.80 if (frame_index // 6) % 2 == 0 else 0.92          # slow pulsing effect
        self._blend_rect(frame, x0, y0, x0 + banner_w, y0 + banner_h, (0, 0, 200), pulse)
        cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + banner_h), (255, 255, 255), 2)

        icon_r = int(18 * ui)
        for cx in (x0 + icon_r + 14, x0 + banner_w - icon_r - 14):
            self._warning_icon(frame, cx, y0 + int(36 * ui), icon_r)

        subject = (f"AROUND PERSON #{alert.track_id}" if alert.event_type == "crowd_incident"
                   else f"PERSON #{alert.track_id}")
        x_left, x_right = x0 + 2 * icon_r + 24, x0 + banner_w - 2 * icon_r - 24
        self._centered_text(frame, alert.title, x_left, x_right, y0 + int(45 * ui), 0.95 * ui, (255, 255, 255), 2)
        self._centered_text(frame, subject, x0, x0 + banner_w, y0 + int(72 * ui), 0.7 * ui, (255, 255, 255), 2)
        self._centered_text(frame, f"CONFIDENCE: {int(round(alert.confidence * 100))}%",
                            x0, x0 + banner_w, y0 + int(96 * ui), 0.6 * ui, (255, 255, 255), 1)
        self._centered_text(frame, "automated visual warning - human review required",
                            x0, x0 + banner_w, y0 + int(112 * ui), 0.4 * ui, (230, 230, 230), 1)

