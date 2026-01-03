import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

KNOWN_FACES_DIR = "data/known_faces"
EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"

os.makedirs("embeddings", exist_ok=True)

print("🔄 Loading InsightFace (ArcFace + SCRFD)...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # CPU
print("✓ Model loaded\n")

all_embeddings = []
all_names = []

total_images = 0
used_images = 0

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"👤 Processing person: {person_name}")

    for img_name in os.listdir(person_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        total_images += 1
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  ⚠️ Failed to read image: {img_name}")
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"  ⚠️ No face detected: {img_name}")
            continue

        # Use the largest face if multiple are detected
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)  # normalize

        all_embeddings.append(embedding)
        all_names.append(person_name)
        used_images += 1

        print(f"  ✓ {img_name}")

print("\n📊 Summary")
print(f"Total images found: {total_images}")
print(f"Images used (faces detected): {used_images}")

if used_images == 0:
    print("❌ No embeddings created. Check your images.")
    exit()

with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump((np.array(all_embeddings), all_names), f)

print(f"\n💾 Embeddings saved to: {EMBEDDINGS_PATH}")
print(f"✓ Total embeddings stored: {len(all_embeddings)}")
