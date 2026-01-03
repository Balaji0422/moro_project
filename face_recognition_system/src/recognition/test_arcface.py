import cv2
from insightface.app import FaceAnalysis

print("🔄 Loading InsightFace (ArcFace + SCRFD)...")

# buffalo_l includes:
# - SCRFD face detector
# - ArcFace face recognition model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # CPU mode

print("✓ InsightFace model loaded")

# Path to test image
img_path = "data/test_images/test.jpeg"

img = cv2.imread(img_path)
if img is None:
    print("❌ Test image not found:", img_path)
    exit()

# Detect faces
faces = app.get(img)

print(f"✓ Faces detected: {len(faces)}")

if len(faces) > 0:
    face = faces[0]
    embedding = face.embedding

    print("✓ Embedding shape:", embedding.shape)
    print("✓ First 10 embedding values:")
    print(embedding[:10])
else:
    print("⚠️ No face detected in the image")
