"""
roi_picker.py - click the escalator polygon (and direction) on a frame of YOUR video.

Run:  python roi_picker.py
      python roi_picker.py --input input/escalator_input.mp4 --frame 200

Mouse : left click adds a point (polygon mode) or sets the direction arrow (direction mode)
Keys  : d = switch polygon/direction mode    u = undo last point    r = reset
        Enter or s = print the config lines    q or Esc = quit
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

import config as cfg

MAX_DISPLAY_WIDTH = 1280


class RoiPicker:
    def __init__(self, frame):
        self.frame = frame
        self.height, self.width = frame.shape[:2]
        self.scale = min(1.0, MAX_DISPLAY_WIDTH / float(self.width))
        self.polygon = []                 # points in ORIGINAL frame pixels
        self.direction_points = []        # [start, end]
        self.mode = "polygon"

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = (int(round(x / self.scale)), int(round(y / self.scale)))   # display -> original
        if self.mode == "polygon":
            self.polygon.append(point)
        else:
            if len(self.direction_points) >= 2:
                self.direction_points = []
            self.direction_points.append(point)

    def _to_display(self, point):
        return int(point[0] * self.scale), int(point[1] * self.scale)

    def render(self):
        canvas = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale) if self.scale < 1.0 \
            else self.frame.copy()
        if len(self.polygon) >= 2:
            pts = np.array([self._to_display(p) for p in self.polygon], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], len(self.polygon) >= 3, (255, 160, 0), 2, cv2.LINE_AA)
        for i, p in enumerate(self.polygon):
            d = self._to_display(p)
            cv2.circle(canvas, d, 5, (0, 255, 255), -1)
            cv2.putText(canvas, str(i + 1), (d[0] + 6, d[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if len(self.direction_points) == 2:
            cv2.arrowedLine(canvas, self._to_display(self.direction_points[0]),
                            self._to_display(self.direction_points[1]), (0, 255, 0), 3,
                            cv2.LINE_AA, tipLength=0.2)
        elif len(self.direction_points) == 1:
            cv2.circle(canvas, self._to_display(self.direction_points[0]), 6, (0, 255, 0), -1)
        help_text = f"mode: {self.mode.upper()} | d=switch u=undo r=reset Enter=print q=quit"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(canvas, help_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def print_config(self):
        print("\n" + "=" * 60)
        print("Paste these lines into config.py (replace the existing ones):\n")
        print(f"REFERENCE_FRAME_SIZE = ({self.width}, {self.height})")
        if len(self.polygon) >= 3:
            print("ESCALATOR_POLYGON = [")
            for x, y in self.polygon:
                print(f"    ({x}, {y}),")
            print("]")
        else:
            print("# (need at least 3 polygon points)")
        if len(self.direction_points) == 2:
            (x1, y1), (x2, y2) = self.direction_points
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length > 1e-6:
                print(f"ESCALATOR_DIRECTION = ({dx / length:.3f}, {dy / length:.3f})")
        else:
            print("# (press d and click start + end of the direction arrow to get ESCALATOR_DIRECTION)")
        print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick the escalator ROI polygon and direction")
    parser.add_argument("--input", default=str(cfg.INPUT_VIDEO))
    parser.add_argument("--frame", type=int, default=0, help="frame index to display")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(Path(args.input)))
    if not capture.isOpened():
        print(f"[ERROR] Cannot open video: {args.input}")
        return 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        print("[ERROR] Could not read a frame from the video.")
        return 1

    picker = RoiPicker(frame)
    window = "ROI picker"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, picker.on_mouse)
    while True:
        cv2.imshow(window, picker.render())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("d"):
            picker.mode = "direction" if picker.mode == "polygon" else "polygon"
        elif key == ord("u"):
            if picker.mode == "polygon" and picker.polygon:
                picker.polygon.pop()
            elif picker.direction_points:
                picker.direction_points.pop()
        elif key == ord("r"):
            picker.polygon.clear()
            picker.direction_points.clear()
        elif key in (13, ord("s")):
            picker.print_config()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
