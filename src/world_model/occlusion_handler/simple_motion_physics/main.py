# =========================================================
# main.py — PURE ReID TRACKING (FINAL)
# =========================================================``

import cv2

from reid import ReIDModel
from tracker import Tracker
from ui import UI
from world import WorldModel




# ---------------------------------------------------------
# ReID matching
# ---------------------------------------------------------
def match_by_reid(detections, frame, reid, world, entity_id, threshold=0.6):

    entity = world.entities.get(entity_id)
    if entity is None:
        return None

    target_emb = world.get_stable_embedding(entity)
    if target_emb is None:
        return None

    best_det = None
    best_score = 0.0
    best_emb = None

    for det in detections:
        x1, y1, x2, y2, _ = det
        crop = frame[y1:y2, x1:x2]

        emb = reid.get_embedding(crop)
        if emb is None:
            continue

        score = reid.similarity(target_emb, emb)

        if score > best_score:
            best_score = score
            best_det = det
            best_emb = emb

    if best_det is not None and best_score >= threshold:
        return best_det, best_emb, best_score

    return None


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = Tracker()
    reid    = ReIDModel()
    world   = WorldModel()
    ui      = UI()

    world.fps = fps

    poi_initialized = False
    poi_id = 1

    print("[INFO] Waiting for person...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = tracker.detect(frame)

        # ============================================
        # AUTO INIT
        # ============================================
        if not poi_initialized:

            if len(detections) > 0:
                best_det = max(
                    detections,
                    key=lambda d: (d[2]-d[0])*(d[3]-d[1])
                )

                x1, y1, x2, y2, _ = best_det
                box = (x1, y1, x2, y2)

                crop = frame[y1:y2, x1:x2]
                emb = reid.get_embedding(crop)

                world.add_poi(poi_id, box, emb)

                print("[INIT] POI initialized")
                poi_initialized = True

            cv2.imshow("World Model", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # ============================================
        # REID TRACKING
        # ============================================
        detected_ids = set()

        if len(detections) > 0:
            match = match_by_reid(
                detections, frame, reid, world, poi_id
            )

            if match is not None:
                det, emb, score = match
                x1, y1, x2, y2, _ = det
                box = (x1, y1, x2, y2)

                print(f"[ReID] matched score={score:.2f}")

                world.update_entity(poi_id, box, emb)
                detected_ids.add(poi_id)

        # ============================================
        # WORLD UPDATE
        # ============================================
        world.tick(detected_ids)

        # ============================================
        # UI
        # ============================================
        display = ui.draw_world(frame, world)
        cv2.imshow("World Model", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = "/Users/manishpoudel/Projects/hexlipi/cv-tryout/world_model/demo_video.mp4"   # <-- change to your video
    main(video_path)