# =========================================================
# world.py — The World Model
#
# Holds everything the system knows or believes about
# the scene at any moment.
#
# REALITY    = what YOLO actually detected
# IMAGINATION = what system predicts during occlusion
# =========================================================

from collections import deque
import time


# =========================
# ENTITY STATES
# =========================

VISIBLE   = "VISIBLE"
OCCLUDED  = "OCCLUDED"
LOST      = "LOST"


# =========================
# WORLD MODEL
# =========================

class WorldModel:

    def __init__(self):

        # All entities in the scene
        # entity_id → entity dict
        self.entities = {}

        # Relations between entity pairs
        # (id_a, id_b) → relation dict
        self.relations = {}

        # Event log — everything that happened
        self.events = []

        self.frame_number = 0
        self.fps = 30.0   # updated when video loads


    # -------------------------------------------------------
    # ADD ENTITIES
    # -------------------------------------------------------

    def add_poi(self, entity_id, box, embedding=None):
        """Register a Person of Interest."""
        cx, cy = self._center(box)
        self.entities[entity_id] = {
            "type":       "person",
            "label":      f"Person {entity_id}",

            # REALITY
            "state":      VISIBLE,
            "box":        box,
            "center":     (cx, cy),
            "speed":      0.0,
            "velocity":   (0.0, 0.0),
            "trajectory": deque(maxlen=20),

            # IMAGINATION
            "occluded_since":     None,
            "imagined_center":    None,
            "imagined_trajectory": deque(maxlen=20),

            # RE-ID
            "embedding":          embedding,
            "embedding_history":  [],

            # VALIDATION
            "imagination_errors": [],   # error (px) each reappearance
        }
        self.entities[entity_id]["trajectory"].append((cx, cy))
        self._log_event(entity_id, "REGISTERED as POI")


    def add_ooi(self, entity_id, box):
        """Register an Object of Interest (static)."""
        cx, cy = self._center(box)
        self.entities[entity_id] = {
            "type":   "object",
            "label":  f"Object {entity_id}",
            "state":  VISIBLE,
            "box":    box,
            "center": (cx, cy),
        }
        self._log_event(entity_id, "REGISTERED as OOI")


    # -------------------------------------------------------
    # UPDATE ENTITY EVERY FRAME (called from tracker)
    # -------------------------------------------------------

    def update_entity(self, entity_id, box, embedding=None):
        """
        Called when YOLO detects this entity.
        Updates REALITY state.
        If it was previously OCCLUDED → validate imagination.
        """
        e = self.entities.get(entity_id)
        if e is None or e["type"] == "object":
            return

        cx, cy = self._center(box)
        prev_center = e["center"]
        prev_state  = e["state"]

        # --- velocity and speed ---
        dt = 1.0 / self.fps
        vx = (cx - prev_center[0]) / dt
        vy = (cy - prev_center[1]) / dt
        speed = (vx**2 + vy**2) ** 0.5

        # smooth speed over trajectory
        e["trajectory"].append((cx, cy))
        smooth_speed = self._smooth_speed(e["trajectory"])

        # --- update REALITY ---
        e["state"]    = VISIBLE
        e["box"]      = box
        e["center"]   = (cx, cy)
        e["velocity"] = (vx, vy)
        e["speed"]    = smooth_speed

        # --- update embedding ---
        if embedding is not None:
            e["embedding_history"].append(embedding)
            if len(e["embedding_history"]) > 10:
                e["embedding_history"].pop(0)
            e["embedding"] = embedding

        # --- VALIDATE imagination if was occluded ---
        if prev_state == OCCLUDED and e["imagined_center"] is not None:
            error = self._dist(e["imagined_center"], (cx, cy))
            e["imagination_errors"].append(error)
            correct = error < 40   # px threshold

            self._log_event(entity_id,
                f"REAPPEARED | imagined={e['imagined_center']} "
                f"actual=({cx},{cy}) "
                f"error={error:.1f}px "
                f"{'✓ CORRECT' if correct else '✗ WRONG'}")

            e["imagined_center"]    = None
            e["occluded_since"]     = None
            e["imagined_trajectory"].clear()

        # --- walking state ---
        e["walking_state"] = self._walking_state(smooth_speed)


    def mark_occluded(self, entity_id):
        """
        Called when YOLO does NOT detect this entity.
        Switches to IMAGINATION mode.
        """
        e = self.entities.get(entity_id)
        if e is None or e["type"] == "object":
            return
        if e["state"] == LOST:
            return

        if e["state"] == VISIBLE:
            # just became occluded
            e["state"]         = OCCLUDED
            e["occluded_since"] = self.frame_number
            self._log_event(entity_id, "OCCLUDED — imagination started")

        # project position forward using last known velocity
        if e["imagined_center"] is None:
            e["imagined_center"] = e["center"]

        dt = 1.0 / self.fps
        ix, iy = e["imagined_center"]
        vx, vy = e["velocity"]
        ix += vx * dt
        #iy += vy * dt
        e["imagined_center"] = (ix, iy)
        e["imagined_trajectory"].append((ix, iy))

        # if occluded too long → mark lost
        frames_gone = self.frame_number - e["occluded_since"]
        if frames_gone > self.fps * 10:   # 10 seconds
            e["state"] = LOST
            self._log_event(entity_id, "LOST — occluded too long")


    # -------------------------------------------------------
    # UPDATE RELATIONS (called every frame after all updates)
    # -------------------------------------------------------

    def update_relations(self):
        """
        Compute relations between every POI and every OOI.
        """
        persons = [eid for eid, e in self.entities.items()
                   if e["type"] == "person"]
        objects  = [eid for eid, e in self.entities.items()
                   if e["type"] == "object"]

        for pid in persons:
            for oid in objects:
                self._update_relation(pid, oid)


    def _update_relation(self, pid, oid):
        person = self.entities[pid]
        obj    = self.entities[oid]

        key = (pid, oid)

        # use imagined center if occluded
        if person["state"] == VISIBLE:
            pcenter = person["center"]
        elif person["state"] == OCCLUDED:
            pcenter = person["imagined_center"] or person["center"]
        else:
            return

        ocenter = obj["center"]

        dist = self._dist(pcenter, ocenter)

        # previous relation
        prev = self.relations.get(key, {})
        prev_dist = prev.get("distance", dist)

        approaching = dist < prev_dist - 2   # moving closer
        leaving     = dist > prev_dist + 2   # moving away

        # reached = bounding boxes overlap or very close
        reached = self._boxes_overlap(person.get("box"), obj["box"]) \
                  or dist < 60

        # dwell time
        if reached:
            if prev.get("dwell_start") is None:
                dwell_start = self.frame_number
            else:
                dwell_start = prev["dwell_start"]
            dwell_frames = self.frame_number - dwell_start
            dwell_seconds = dwell_frames / self.fps
        else:
            dwell_start   = None
            dwell_seconds = 0.0

        # direction vector from person toward OOI
        dx = ocenter[0] - pcenter[0]
        dy = ocenter[1] - pcenter[1]

        # velocity dot with direction → positive = approaching
        vx, vy = person.get("velocity", (0, 0))
        mag = (dx**2 + dy**2) ** 0.5 + 1e-6
        dot = (vx * dx + vy * dy) / mag

        relation = {
            "distance":      dist,
            "approaching":   dot > 5,    # moving toward OOI
            "leaving":       dot < -5,   # moving away
            "reached":       reached,
            "dwell_start":   dwell_start,
            "dwell_seconds": dwell_seconds,
        }

        # fire events on state changes
        if reached and not prev.get("reached"):
            self._log_event(pid, f"REACHED {self.entities[oid]['label']}")

        if not reached and prev.get("reached"):
            ds = prev.get("dwell_seconds", 0)
            self._log_event(pid,
                f"LEFT {self.entities[oid]['label']} "
                f"after {ds:.1f}s")

        if relation["approaching"] and not prev.get("approaching"):
            self._log_event(pid,
                f"APPROACHING {self.entities[oid]['label']} "
                f"dist={dist:.0f}px")

        self.relations[key] = relation


    # -------------------------------------------------------
    # TICK — called every frame
    # -------------------------------------------------------

    def tick(self, detected_ids):
        """
        detected_ids: set of entity IDs YOLO found this frame.
        Mark everything else as occluded.
        """
        self.frame_number += 1

        for eid, e in self.entities.items():
            if e["type"] == "object":
                continue
            if eid not in detected_ids:
                self.mark_occluded(eid)

        self.update_relations()


    # -------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------

    def _center(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _dist(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def _boxes_overlap(self, a, b):
        if a is None or b is None:
            return False
        return not (a[2] < b[0] or a[0] > b[2] or
                    a[3] < b[1] or a[1] > b[3])

    def _smooth_speed(self, trajectory):
        pts = list(trajectory)
        if len(pts) < 2:
            return 0.0
        total = sum(self._dist(pts[i], pts[i-1])
                    for i in range(1, len(pts)))
        return (total / (len(pts) - 1)) * self.fps

    def _walking_state(self, speed):
        if speed < 5:
            return "STILL"
        elif speed < 30:
            return "SLOW"
        elif speed < 80:
            return "WALKING"
        else:
            return "FAST"

    def _log_event(self, entity_id, description):
        t = self.frame_number / max(self.fps, 1)
        event = {
            "frame":  self.frame_number,
            "time":   t,
            "entity": entity_id,
            "desc":   description,
        }
        self.events.append(event)
        label = self.entities.get(entity_id, {}).get("label", entity_id)
        print(f"[t={t:.1f}s] {label}: {description}")

    def get_recent_events(self, n=10):
        return self.events[-n:]
    
    def get_stable_embedding(self, entity):
        history = entity.get("embedding_history", [])
        if not history:
            return entity.get("embedding")

        emb = sum(history) / len(history)
        norm = (emb**2).sum() ** 0.5 + 1e-6
        return emb / norm
