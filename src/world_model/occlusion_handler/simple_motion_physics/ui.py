# =========================================================
# ui.py — User Interface
#
# 1. First frame: user draws POI and OOI boxes
# 2. Live overlay: show world state on every frame
# =========================================================

import cv2
import numpy as np


# Colors
GREEN  = (0, 220, 80)     # POI visible
ORANGE = (0, 165, 255)    # POI occluded
RED    = (0, 60, 220)     # POI lost
BLUE   = (220, 160, 0)    # OOI
YELLOW = (0, 220, 220)    # imagined position
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)
BLACK  = (0, 0, 0)


class UI:

    def __init__(self):
        self.drawing     = False
        self.start_point = None
        self.end_point   = None
        self.boxes       = []        # list of (box, label)
        self.current_box = None
        self._mode       = None      # "POI" or "OOI"
        self._done       = False

    # -------------------------------------------------------
    # FIRST FRAME — draw boxes interactively
    # -------------------------------------------------------

    def select_entities(self, frame):
        """
        Shows first frame. User draws:
          - One box for POI (press P first)
          - One or more boxes for OOI (press O first)
          - Press ENTER when done
        Returns list of (box, type) where type is "POI" or "OOI"
        """
        clone = frame.copy()
        self.boxes = []
        self._mode = None
        self._done = False

        window = "Setup — [P] Person  [O] Object  [ENTER] Done  [Z] Undo"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._mouse_cb)

        instructions = [
            "Press P then draw box around PERSON",
            "Press O then draw box around OBJECT/SHELF",
            "Press ENTER when done",
            "Press Z to undo last box",
        ]

        while True:
            display = clone.copy()

            # draw committed boxes
            for box, btype in self.boxes:
                color = GREEN if btype == "POI" else BLUE
                x1, y1, x2, y2 = box
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, btype, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # draw in-progress box
            if self.drawing and self.start_point and self.end_point:
                color = GREEN if self._mode == "POI" else BLUE
                cv2.rectangle(display,
                              self.start_point, self.end_point,
                              color, 1)

            # mode indicator
            mode_text = f"Mode: {self._mode or 'none — press P or O'}"
            cv2.rectangle(display, (0, 0), (500, 30), BLACK, -1)
            cv2.putText(display, mode_text, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

            # instructions
            for i, line in enumerate(instructions):
                cv2.putText(display, line,
                            (8, frame.shape[0] - 20 - i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('p') or key == ord('P'):
                self._mode = "POI"
            elif key == ord('o') or key == ord('O'):
                self._mode = "OOI"
            elif key == ord('z') or key == ord('Z'):
                if self.boxes:
                    self.boxes.pop()
            elif key == 13:   # ENTER
                if self.boxes:
                    break

        cv2.destroyWindow(window)
        return self.boxes

    def _mouse_cb(self, event, x, y, flags, param):
        if self._mode is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing     = True
            self.start_point = (x, y)
            self.end_point   = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing   = False
            self.end_point = (x, y)

            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.boxes.append(((x1, y1, x2, y2), self._mode))

    # -------------------------------------------------------
    # LIVE OVERLAY — draw world state on each frame
    # -------------------------------------------------------

    def draw_world(self, frame, world):
        overlay = frame.copy()
        h, w    = frame.shape[:2]

        for eid, entity in world.entities.items():

            # -----------------------------------------------
            # DRAW OBJECTS (OOI) — static blue box
            # -----------------------------------------------
            if entity["type"] == "object":
                x1, y1, x2, y2 = entity["box"]
                cv2.rectangle(overlay, (x1,y1), (x2,y2), BLUE, 2)
                cv2.putText(overlay, entity["label"],
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)
                continue

            # -----------------------------------------------
            # DRAW PERSONS (POI)
            # -----------------------------------------------
            state = entity["state"]

            if state == "VISIBLE":
                color = GREEN
                x1, y1, x2, y2 = entity["box"]
                cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)

                # label: ID + walking state + speed
                ws    = entity.get("walking_state", "")
                speed = entity.get("speed", 0)
                label = f"{entity['label']} | {ws} {speed:.0f}px/s"
                cv2.putText(overlay, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # draw trajectory
                traj = list(entity["trajectory"])
                for i in range(1, len(traj)):
                    cv2.line(overlay, traj[i-1], traj[i],
                             GREEN, 1)

            elif state == "OCCLUDED":
                color = ORANGE

                # draw imagined position as dot + circle
                ic = entity.get("imagined_center")
                if ic:
                    ix, iy = int(ic[0]), int(ic[1])
                    cv2.circle(overlay, (ix, iy), 14, ORANGE, 2)
                    cv2.circle(overlay, (ix, iy), 4,  ORANGE, -1)
                    cv2.putText(overlay,
                                f"{entity['label']} (img)",
                                (ix + 16, iy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, ORANGE, 2)

                    # draw imagined trajectory dashed
                    itraj = list(entity["imagined_trajectory"])
                    for i in range(1, len(itraj)):
                        pt1 = (int(itraj[i-1][0]), int(itraj[i-1][1]))
                        pt2 = (int(itraj[i][0]),   int(itraj[i][1]))
                        cv2.line(overlay, pt1, pt2, YELLOW, 1)

            elif state == "LOST":
                pass   # don't draw lost entities

            # -----------------------------------------------
            # DRAW RELATIONS
            # -----------------------------------------------
            for (pid, oid), rel in world.relations.items():
                if pid != eid:
                    continue

                person = world.entities[pid]
                obj    = world.entities[oid]

                # use imagined center if occluded
                if person["state"] == "VISIBLE":
                    pcenter = person["center"]
                else:
                    pcenter = person.get("imagined_center") or person["center"]

                ocenter = obj["center"]
                pcenter = (int(pcenter[0]), int(pcenter[1]))
                ocenter = (int(ocenter[0]), int(ocenter[1]))

                # line color based on relation
                if rel.get("reached"):
                    line_color = GREEN
                    thickness  = 2
                elif rel.get("approaching"):
                    line_color = YELLOW
                    thickness  = 1
                else:
                    line_color = GRAY
                    thickness  = 1

                cv2.line(overlay, pcenter, ocenter, line_color, thickness)

                # distance label on midpoint
                mid = ((pcenter[0]+ocenter[0])//2,
                       (pcenter[1]+ocenter[1])//2)
                dist = rel.get("distance", 0)
                cv2.putText(overlay, f"{dist:.0f}px",
                            mid, cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, line_color, 1)

        # -----------------------------------------------
        # SIDE PANEL — world state text
        # -----------------------------------------------
        panel_w = 280
        panel   = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)

        y_offset = 20
        def put(text, color=WHITE, scale=0.5, bold=False):
            nonlocal y_offset
            thickness = 2 if bold else 1
            cv2.putText(panel, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness)
            y_offset += int(scale * 36)

        put("WORLD MODEL", WHITE, 0.6, bold=True)
        put(f"Frame {world.frame_number}", GRAY, 0.45)
        put("", WHITE, 0.3)

        # entity states
        for eid, entity in world.entities.items():
            if entity["type"] == "object":
                put(entity["label"], BLUE, 0.5, bold=True)
                put("  [FIXED]", GRAY, 0.42)
            else:
                state = entity["state"]
                sc = GREEN if state=="VISIBLE" else \
                     ORANGE if state=="OCCLUDED" else RED
                put(entity["label"], sc, 0.5, bold=True)
                put(f"  {state}", sc, 0.42)

                if state == "VISIBLE":
                    ws    = entity.get("walking_state", "")
                    speed = entity.get("speed", 0)
                    put(f"  {ws} {speed:.0f} px/s", WHITE, 0.42)

                elif state == "OCCLUDED":
                    frames_gone = (world.frame_number -
                                   (entity.get("occluded_since") or
                                    world.frame_number))
                    secs = frames_gone / max(world.fps, 1)
                    put(f"  occluded {secs:.1f}s", ORANGE, 0.42)
                    ic = entity.get("imagined_center")
                    if ic:
                        put(f"  img pos ({int(ic[0])},{int(ic[1])})",
                            YELLOW, 0.42)

            # relations for this entity
            for (pid, oid), rel in world.relations.items():
                if pid != eid:
                    continue
                oname = world.entities[oid]["label"]
                dist  = rel.get("distance", 0)
                dwell = rel.get("dwell_seconds", 0)

                put(f"  → {oname}", GRAY, 0.42)
                put(f"    dist {dist:.0f}px", WHITE, 0.4)

                if rel.get("reached"):
                    put(f"    NEAR  dwell {dwell:.1f}s", GREEN, 0.42)
                elif rel.get("approaching"):
                    put(f"    approaching", YELLOW, 0.42)
                else:
                    put(f"    moving away", GRAY, 0.42)

            put("", WHITE, 0.3)

        # imagination accuracy summary
        put("IMAGINATION", WHITE, 0.5, bold=True)
        for eid, entity in world.entities.items():
            if entity["type"] != "person":
                continue
            errs = entity.get("imagination_errors", [])
            if errs:
                avg = sum(errs) / len(errs)
                put(f"  {entity['label']}", WHITE, 0.42)
                put(f"  avg error {avg:.1f}px", GREEN, 0.42)

        put("", WHITE, 0.3)

        # recent events
        put("EVENTS", WHITE, 0.5, bold=True)
        events = world.get_recent_events(6)
        for ev in reversed(events):
            t    = ev["time"]
            desc = ev["desc"][:28]
            put(f"  {t:.1f}s {desc}", GRAY, 0.38)

        # blend overlay
        result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        # attach panel
        if result.shape[0] != panel.shape[0]:
            panel = cv2.resize(panel, (panel_w, result.shape[0]))

        return np.hstack([result, panel])
