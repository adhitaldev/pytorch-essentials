# =========================================================
# tracker.py — YOLO Detection + IoU Matching
#
# Each frame:
# 1. Run YOLO on frame
# 2. Match detections to known POI using IoU
# 3. Return matched/unmatched detections
# =========================================================

import numpy as np
from ultralytics import YOLO


class Tracker:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        print("[Tracker] YOLO loaded.")

    def detect(self, frame, conf=0.5):
        """
        Run YOLO on frame.
        Returns list of boxes for class=person only.
        Each box: (x1, y1, x2, y2, confidence)
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            if int(box.cls[0]) != 0:        # person only
                continue
            conf_score = float(box.conf[0])
            if conf_score < conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # quality checks
            bw = x2 - x1
            bh = y2 - y1
            if bw < 30 or bh < 60:
                continue
            aspect = bh / (bw + 1e-6)
            if aspect < 1.2 or aspect > 5.0:
                continue

            detections.append((x1, y1, x2, y2, conf_score))

        return detections

    def match_to_entity(self, detections, entity_box, threshold=0.15):
        """
        Find which detection best matches a known entity box.
        Returns (best_detection, iou_score) or (None, 0)
        """
        best_det   = None
        best_score = 0.0

        for det in detections:
            score = self._iou(entity_box, det[:4])
            if score > best_score:
                best_score = score
                best_det   = det

        if best_score >= threshold:
            return best_det, best_score
        return None, 0.0

    def _iou(self, a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2]-a[0]) * (a[3]-a[1])
        areaB = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)
