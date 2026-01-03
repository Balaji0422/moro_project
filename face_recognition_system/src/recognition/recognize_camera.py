import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ================== CONFIG ==================
FRAME_SKIP = 3          # 🔥 increase for more speed (2–4)
THRESHOLD = 0.45
EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"
# ============================================

print("🔄 Loading InsightFace (SCRFD + ArcFace)...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(416, 416))  # 🔥 smaller detector input
print("✓ InsightFace ready")

print("🔄 Loading known face embeddings...")
with open(EMBEDDINGS_PATH, "rb") as f:
    known_embeddings, known_names = pickle.load(f)
known_embeddings = np.array(known_embeddings)
print(f"✓ Loaded {len(known_names)} embeddings")

print("\n📷 Opening camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✓ Camera opened | Press Q to quit\n")

frame_id = 0
last_faces = []   # 🔥 cache results

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # ========= RUN HEAVY MODELS ONLY ON SKIPPED FRAMES =========
    if frame_id % FRAME_SKIP == 0:
        faces = app.get(frame)
        last_faces = []

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            sims = cosine_similarity(
                embedding.reshape(1, -1), known_embeddings
            )[0]

            best_idx = np.argmax(sims)
            best_score = sims[best_idx]

            if best_score > THRESHOLD:
                name = known_names[best_idx]
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            last_faces.append((x1, y1, x2, y2, name, best_score, color))

    # ========= DRAW LAST KNOWN RESULTS (FAST) =========
    for (x1, y1, x2, y2, name, score, color) in last_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} ({score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Live Face Recognition (Optimized)", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
print("✓ Camera closed")
