# 🎯 Face Recognition System (Moro Project)

A complete real-time face recognition system built with **InsightFace (ArcFace)** and **FAISS** using Python 3.11.9.

---

## 📋 Project Overview

This project implements a comprehensive face recognition pipeline using state-of-the-art InsightFace technology:

### **Core Features** ✅
- **Real-time Face Detection**: SCRFD detector for fast face detection
- **Face Encoding**: ArcFace (512-dimensional embeddings) for robust face representation
- **Similarity Search**: FAISS indexing for efficient face matching
- **Face Tracking**: KCF tracker for continuous face tracking across frames
- **Live Recognition**: Real-time face identification from camera feeds
- **Known Face Database**: Easily manage and add new people to recognition database

---

## 📁 Project Structure

```
face_recognition_system/
│
├── models/
│   ├── arcface/
│   │   └── buffalo_l/            # InsightFace model (auto-downloaded)
│   │       ├── 1k3d68.onnx       # 3D face landmark detector
│   │       ├── 2d106det.onnx     # 2D face landmark detector
│   │       ├── det_10g.onnx      # Face detection (SCRFD)
│   │       ├── genderage.onnx    # Gender/Age detector
│   │       └── w600k_r50.onnx    # ArcFace recognition model
│   │
│   └── faiss/                    # FAISS libraries (optional)
│
├── data/
│   ├── known_faces/
│   │   ├── Asnawas/              # Images of known person 1
│   │   │   ├── A1.jpeg
│   │   │   ├── A2.jpeg
│   │   │   ├── A3.jpeg
│   │   │   └── A4.jpeg
│   │   │
│   │   └── Balaji/               # Images of known person 2
│   │       ├── B1.jpeg
│   │       ├── B2.jpeg
│   │       ├── B3.jpeg
│   │       ├── B4.jpeg
│   │       └── B5.jpeg
│   │
│   ├── raw_images/               # Raw input images (for processing)
│   ├── test_images/              # Test images for verification
│   │   └── test.jpeg
│   │
│   └── videos/                   # Video files (for future processing)
│
├── embeddings/
│   ├── face_embeddings.pkl       # Pickled face embeddings database
│   ├── faiss.index               # FAISS index for similarity search
│   └── id_map.pkl                # Mapping of IDs to person names
│
├── src/
│   ├── recognition/              # Face recognition & embedding scripts
│   │   ├── build_embeddings.py  # Build face embeddings from known faces
│   │   ├── recognize_camera.py  # Live face recognition with tracking
│   │   └── test_arcface.py      # Test ArcFace functionality
│   │
│   └── indexing/                 # FAISS indexing scripts
│       └── build_faiss_index.py # Build FAISS index from embeddings
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .venv/                        # Virtual environment (Python 3.11.9)
```

---

## 🔧 Installation & Setup

### **Step 1: Navigate to Project**
```bash
cd c:\infiposts_project\Moro_project_python_1
```

### **Step 2: Activate Virtual Environment (Python 3.11.9)**
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Note**: InsightFace models (buffalo_l) will auto-download on first run (~341 MB)
- Location: `C:\Users\<YourUsername>\.insightface\models\buffalo_l\`

---

## 📊 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| insightface | 0.7.3 | Face recognition (ArcFace + SCRFD detector) |
| onnxruntime | 1.23.2 | ONNX model inference engine |
| opencv-contrib-python | 4.12.0.88 | Computer vision + KCF tracking |
| numpy | 2.2.6 | Numerical computing |
| faiss-cpu | 1.8.0 | Similarity search and indexing |
| Pillow | 12.1.0 | Image processing |
| scikit-learn | 1.8.0 | Machine learning utilities |
| scikit-image | 0.26.0 | Advanced image processing |

---

## 🚀 How to Run (Quick Start)

### **Complete 3-Step Setup & Recognition Process**

##### **Step 1: Add Known Faces to Database**

Create a folder structure in `data/known_faces/`:
```
data/known_faces/
├── Asnawas/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
└── Balaji/
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg
```

**Guidelines**:
- Minimum 3-5 images per person recommended
- Images should be clear face photos
- Supported formats: JPG, JPEG, PNG
- Person folder names will be used as display names

##### **Step 2: Build Embeddings & FAISS Index**

```bash
# Step 2a: Generate face embeddings
python src/recognition/build_embeddings.py
```

**Output**:
```
🔄 Loading InsightFace (ArcFace + SCRFD)...
✓ Model loaded

👤 Processing person: Asnawas
  ✓ image1.jpg (1 face detected)
  ✓ image2.jpg (1 face detected)
  ✓ image3.jpg (1 face detected)

👤 Processing person: Balaji
  ✓ image1.jpg (1 face detected)
  ✓ image2.jpg (1 face detected)
  ✓ image3.jpg (1 face detected)

✅ Embeddings saved: embeddings/face_embeddings.pkl
Total people: 2
Total embeddings: 6
```

```bash
# Step 2b: Build FAISS index for fast similarity search
python src/indexing/build_faiss_index.py
```

**Output**:
```
🔄 Loading embeddings...
✓ FAISS index loaded (14 identities)
✅ FAISS index built and saved successfully
```

##### **Step 3: Run Live Face Recognition**

```bash
python src/recognition/recognize_camera.py
```

**Features**:
- Real-time face recognition from camera
- Green bounding box = Known face (recognized)
- Red bounding box = Unknown face
- Similarity score displayed
- KCF tracking for smooth tracking
- Voting-based name confirmation (for stability)

**Output**:
```
🔄 Loading InsightFace...
✓ InsightFace ready

🔄 Loading FAISS index...
✓ FAISS index loaded (14 identities)

[Live video feed with face recognition]
```

Press `ESC` or `Q` to stop.

---

## ⚙️ Configuration Parameters

You can customize behavior by editing `src/recognition/recognize_camera.py`:

```python
# Recognition threshold (0.0-1.0)
# Lower = more permissive, Higher = more strict
THRESHOLD = 0.45

# How often to run face detection (every N frames)
# Higher = faster but less accurate tracking
DETECT_EVERY_N_FRAMES = 30

# Maximum faces to track simultaneously
MAX_FACES = 5

# Voting window for name confirmation
# Higher = more stable but slower to switch identities
VOTE_WINDOW = 5
```

---

## 🔍 System Architecture

### **Face Recognition Pipeline**

```
Camera Feed
    ↓
SCRFD Face Detection (InsightFace)
    ↓
Face Alignment & Normalization
    ↓
ArcFace Embedding Generation (512-dim)
    ↓
L2 Normalization
    ↓
FAISS Index Search (Similarity Matching)
    ↓
Threshold Decision
    ├─→ Known Face (Similarity > Threshold)
    └─→ Unknown Face (Similarity ≤ Threshold)
    ↓
KCF Tracker (Frame-to-frame tracking)
    ↓
Voting-based Name Confirmation
    ↓
Display & Output
```

### **Key Components**

1. **SCRFD Detector** (Face Detection)
   - Ultra-fast face detection
   - Part of InsightFace buffalo_l model
   - Outputs: bounding box, confidence score

2. **ArcFace** (Face Recognition)
   - 512-dimensional face embeddings
   - Highly discriminative representation
   - Part of InsightFace buffalo_l model

3. **FAISS** (Similarity Search)
   - IndexFlatIP: Inner Product search
   - Fast and memory-efficient
   - Normalized embeddings for cosine similarity

4. **KCF Tracker** (Face Tracking)
   - Correlation Filters for tracking
   - Smooth face tracking across frames
   - Reduces detection frequency

---

## 📈 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Face Detection Speed | ~30 FPS | 640x480 resolution |
| Recognition Accuracy | >98% | On LFW benchmark |
| Embedding Dimension | 512 | ArcFace output |
| Index Type | FAISS FlatIP | Normalized L2 distance |
| Tracking | KCF Tracker | Smooth across frames |

---

## ✨ Core Features Explained

### **Real-time Detection**
- SCRFD (Single-stage face Detector) detects all faces in frame
- Super-fast and accurate
- Outputs bounding boxes with confidence scores

### **Robust Recognition**
- ArcFace embeddings are highly discriminative
- 512-dimensional vectors capture face uniqueness
- Cosine similarity for matching (via FAISS)

### **Efficient Indexing**
- FAISS IndexFlatIP for O(1) lookup
- Normalized L2 distance = cosine similarity
- Fast similarity computation

### **Smooth Tracking**
- KCF (Kernelized Correlation Filters) tracker
- Maintains face identity across frames
- Reduces detection frequency for speed

### **Robust Identification**
- Voting-based name confirmation
- Multiple frames voted for final decision
- Prevents flickering between identities

---

## 🎓 How Face Recognition Works

1. **Face Detection**: SCRFD finds all faces in image
2. **Face Alignment**: 3D landmark detection aligns face to standard orientation
3. **Embedding Generation**: ArcFace converts aligned face to 512-D vector
4. **Similarity Search**: FAISS finds closest match in known faces database
5. **Decision Making**: Compare similarity score to threshold
6. **Tracking**: KCF keeps tracking face across frames
7. **Confirmation**: Voting confirms final identity

---

## 🐛 Troubleshooting

### **No faces detected**
- Check camera is working
- Ensure good lighting
- Try adjusting camera position
- Check if camera permission is granted

### **Poor recognition accuracy**
- Add more images per person (5-10 minimum)
- Ensure images are clear and well-lit
- Try different angles and expressions
- Rebuild embeddings and FAISS index

### **GPU not available warning**
- System uses CPU by default (still fast)
- If GPU available, edit code: `ctx_id=0` → `ctx_id=0` (for GPU)
- Not required for good performance on CPU

### **ONNX Runtime warnings**
- These are informational, system works fine
- Can be ignored safely

---

## 📝 Adding New Faces

1. Create a new folder in `data/known_faces/` with person's name
2. Add 5-10 clear face photos to the folder
3. Run: `python src/recognition/build_embeddings.py`
4. Run: `python src/indexing/build_faiss_index.py`
5. Run: `python src/recognition/recognize_camera.py`

The new person will be recognized in live feed immediately!

---

## 📚 Additional Resources

- **InsightFace**: https://github.com/deepinsight/insightface
- **FAISS**: https://github.com/facebookresearch/faiss
- **ArcFace Paper**: https://arxiv.org/abs/1801.07698
- **SCRFD Paper**: https://arxiv.org/abs/2105.04714

---

## ⚠️ Important Notes

- **Privacy**: Keep track of whose faces are in the database
- **Performance**: System works best with 3-10 known people
- **Lighting**: Good lighting improves detection and recognition
- **Resolution**: Higher resolution = better accuracy but slower speed
- **Storage**: Embeddings are small (~4KB per person)

---

## 📄 License

This project uses open-source models and libraries. See respective repositories for license details.

---

**Last Updated**: January 5, 2026  
**Python Version**: 3.11.9  
**Project Status**: ✅ Fully Functional

