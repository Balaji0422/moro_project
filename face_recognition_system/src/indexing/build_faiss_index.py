# src/indexing/build_faiss_index.py

import pickle
import faiss
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "face_embeddings.pkl")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "faiss.index")
ID_MAP_PATH = os.path.join(BASE_DIR, "embeddings", "id_map.pkl")

print("🔄 Loading embeddings...")
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings, names = pickle.load(f)

embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)

with open(ID_MAP_PATH, "wb") as f:
    pickle.dump(names, f)

print("✅ FAISS index built and saved successfully")
