import cv2
import os
import pickle
import faiss
import numpy as np
from insightface.app import FaceAnalysis
from collections import deque, Counter

# ================== PATH SETUP ==================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")

FAISS_INDEX_PATH = os.path.join(EMBED_DIR, "faiss.index")
ID_MAP_PATH = os.path.join(EMBED_DIR, "id_map.pkl")

# ================== CONFIG ==================
THRESHOLD = 0.45
DETECT_EVERY_N_FRAMES = 30
MAX_FACES = 5
VOTE_WINDOW = 5
# ============================================

print("🔄 Loading InsightFace...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(320, 320))
print("✓ InsightFace ready")

print("🔄 Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

with open(ID_MAP_PATH, "rb") as f:
    id_names = pickle.load(f)

print(f"✓ FAISS index loaded ({index.ntotal} identities)")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracked_faces = []
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # ================= TRACKING =================
    active_tracks = []
    for data in tracked_faces:
        ok, bbox = data["tracker"].update(frame)
        if ok:
            x, y, w, h = map(int, bbox)
            active_tracks.append(data)
            cv2.rectangle(frame, (x, y), (x + w, y + h), data["color"], 2)
            cv2.putText(
                frame,
                data["name"],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                data["color"],
                2
            )

    tracked_faces = active_tracks

    # ================= DETECTION + FAISS =================
    if frame_id % DETECT_EVERY_N_FRAMES == 0 or len(tracked_faces) == 0:
        faces = app.get(frame)[:MAX_FACES]
        tracked_faces = []

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            w, h = x2 - x1, y2 - y1

            emb = face.embedding.astype("float32")
            emb = emb / np.linalg.norm(emb)
            emb = emb.reshape(1, -1)

            D, I = index.search(emb, k=1)
            score = float(D[0][0])
            idx = int(I[0][0])

            if score > THRESHOLD:
                vote = id_names[idx]
            else:
                vote = "Unknown"

            votes = deque(maxlen=VOTE_WINDOW)
            votes.append(vote)

            final_name = Counter(votes).most_common(1)[0][0]
            color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)

            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x1, y1, w, h))

            tracked_faces.append({
                "tracker": tracker,
                "votes": votes,
                "name": final_name,
                "color": color
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{final_name} ({score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    cv2.imshow("Face Recognition (FAISS + Tracking)", frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
