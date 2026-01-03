# 🎯 Face Recognition System (Moro Project)

A complete real-time face detection and recognition system built with **YOLO** (YOLOv8) and **InsightFace (ArcFace)** using Python 3.11.9.

---

## 📋 Project Overview

This project implements a two-phase face recognition pipeline:

### **Phase-1: YOLO Face Detection** ✅
- Real-time face detection using YOLOv8n
- Optimized for speed and accuracy
- Live camera feed processing
- Video recording with detections

### **Phase-2: InsightFace (ArcFace) Recognition** ✅
- Face encoding using ArcFace (512-dim embeddings)
- Known face database management
- Live face recognition and matching
- Face tracking with KCF tracker
- Similarity-based face identification

---

## 📁 Project Structure

```
face_recognition_system/
│
├── models/
│   ├── yolo/
│   │   ├── yolov8n-face.pt      # Face detection model
│   │   └── yolov8n.pt           # General object detection
│   │
│   └── arcface/
│       └── buffalo_l/            # InsightFace model (auto-downloaded)
│           ├── 1k3d68.onnx
│           ├── 2d106det.onnx
│           ├── det_10g.onnx
│           ├── genderage.onnx
│           └── w600k_r50.onnx
│
├── data/
│   ├── known_faces/
│   │   ├── person1/              # Images of known person 1
│   │   ├── person2/              # Images of known person 2
│   │   └── ...
│   │
│   ├── raw_images/               # Raw input images
│   ├── test_images/              # Test images for recognition
│   └── videos/                   # Video files
│
├── embeddings/
│   └── face_embeddings.pkl       # Saved face embeddings database
│
├── outputs/
│   ├── detections/               # Detection results/videos
│   ├── results/                  # Recognition results
│   └── logs/                     # Log files
│
├── src/
│   ├── detection/
│   │   ├── main.py              # YOLO face detection script
│   │   ├── yolo_detector.py     # YOLO detector class
│   │   └── live_camera.py       # Live camera detection
│   │
│   ├── recognition/
│   │   ├── face_encoder.py      # ArcFace encoder
│   │   ├── build_embeddings.py  # Build embeddings database
│   │   ├── recognize_camera.py  # Live face recognition
│   │   └── test_arcface.py      # ArcFace test script
│   │
│   ├── indexing/                # FAISS indexing (optional)
│   └── utils/                   # Utility functions
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔧 Installation & Setup

### **Step 1: Clone/Setup Project**
```bash
cd c:\infiposts_project\Moro_project_python_1
```

### **Step 2: Create Virtual Environment (Python 3.11.9)**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # On Windows PowerShell
```

### **Step 3: Install Dependencies**
```bash
# Uninstall conflicting packages
pip uninstall opencv-python opencv-python-headless -y

# Install all required packages
pip install -r requirements.txt

# OR install manually
pip install insightface onnxruntime opencv-contrib-python numpy
pip install ultralytics torch torchvision  # For YOLO
pip install pillow scikit-learn scikit-image  # Additional deps
```

### **Step 4: Download Models**
Models will auto-download on first run:
- **YOLO models**: Downloaded automatically from Ultralytics
- **ArcFace (buffalo_l)**: Auto-downloaded from GitHub (281MB)

---

## 📊 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| insightface | 0.7.3 | Face recognition (ArcFace + SCRFD) |
| onnxruntime | 1.23.2 | ONNX model inference |
| opencv-contrib-python | 4.12.0.88 | Computer vision + tracking |
| numpy | 2.2.6 | Numerical computing |
| ultralytics | Latest | YOLO object detection |
| torch | Latest | Deep learning framework |
| pillow | 12.1.0 | Image processing |
| scikit-learn | 1.8.0 | Machine learning utilities |
| scikit-image | 0.26.0 | Image processing |

---

## 🚀 How to Run

### **Phase-1: YOLO Face Detection**

#### 1️⃣ Real-time Camera Detection
```bash
cd face_recognition_system
python src/detection/main.py
```
**Output**: 
- Live video display with face bounding boxes
- Saved video: `outputs/detections/detection_YYYYMMDD_HHMMSS.mp4`

#### 2️⃣ Test Detection
```bash
python src/detection/yolo_detector.py
```

---

### **Phase-2: InsightFace Face Recognition**

#### 1️⃣ Test ArcFace Model
```bash
cd face_recognition_system
python src/recognition/test_arcface.py
```
**Output**: Loads model, detects faces, generates 512-dim embeddings

#### 2️⃣ Build Known Face Database
```bash
python src/recognition/build_embeddings.py
```
**Process**:
- Scans `data/known_faces/` directory
- Detects faces in each image
- Generates ArcFace embeddings (512-dim vectors)
- Saves to `embeddings/face_embeddings.pkl`

**Example Output**:
```
👤 Processing person: Asnawas
  ✓ A1.jpeg
  ✓ A2.jpeg
  ✓ A3.jpeg
👤 Processing person: Balaji
  ✓ B1.jpeg
  ✓ B2.jpeg

💾 Embeddings saved to: embeddings/face_embeddings.pkl
✓ Total embeddings stored: 8
```

#### 3️⃣ Live Face Recognition
```bash
python src/recognition/recognize_camera.py
```
**Features**:
- Real-time face detection using SCRFD
- Face matching against known embeddings
- Similarity score display
- Face tracking (KCF tracker)
- Press 'Q' to quit

**Example Output**:
```
🔄 Loading InsightFace (SCRFD + ArcFace)...
✓ InsightFace ready
🔄 Loading known face embeddings...
✓ Loaded 8 embeddings

📷 Opening camera...
✓ Camera opened | Press Q to quit
```

---

## 📝 Workflow

### **For New Users:**

1. **Add Known Faces**
   ```bash
   # Create directories for each person
   mkdir data/known_faces/person_name
   
   # Add 3-5 images per person (face visible, different angles)
   # Copy images to: data/known_faces/person_name/
   ```

2. **Build Embeddings Database**
   ```bash
   python src/recognition/build_embeddings.py
   ```
   This processes all images and creates the recognition database.

3. **Run Live Recognition**
   ```bash
   python src/recognition/recognize_camera.py
   ```
   System will recognize and display names in real-time.

---

## 🎛️ Configuration

### **Camera Settings** (in `src/recognition/recognize_camera.py`)
```python
# Frame skip interval
FRAME_SKIP = 2  # Process every 2nd frame for speed

# Similarity threshold
SIMILARITY_THRESHOLD = 0.6  # Match confidence (0-1)

# Detection confidence
CONF_THRESHOLD = 0.5

# Max face distance
MAX_FACE_DISTANCE = 200
```

### **Model Settings** (in `src/recognition/recognize_camera.py`)
```python
# ArcFace model
MODEL_NAME = "buffalo_l"  # High accuracy model

# Detection size
DET_SIZE = (416, 416)  # Detection input size
```

---

## 🎯 Features

### **Detection Features**
- ✅ Real-time face detection (YOLOv8)
- ✅ Multi-face detection per frame
- ✅ GPU acceleration (CUDA) support
- ✅ Video output with detections
- ✅ FPS monitoring

### **Recognition Features**
- ✅ ArcFace embeddings (512-dim)
- ✅ Cosine similarity matching
- ✅ Unknown face detection
- ✅ Face tracking across frames
- ✅ Multiple person database support
- ✅ Confidence scores display

### **System Features**
- ✅ Auto-model download
- ✅ CPU/GPU support
- ✅ Efficient frame processing
- ✅ Real-time performance
- ✅ Easy database management

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Load Time | ~10 seconds |
| Face Detection | ~30-50 FPS (with YOLO) |
| Face Recognition | ~10-20 FPS (with ArcFace) |
| Embedding Size | 512 dimensions |
| Similarity Algorithm | Cosine Distance |

---

## 🛠️ Troubleshooting

### **Issue: OpenCV Display Error**
```
AttributeError: module 'cv2' has no attribute 'TrackerKCF_create'
```
**Solution**:
```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python
```

### **Issue: Model Not Found**
**Solution**: Models auto-download on first run. Check internet connection.

### **Issue: Camera Not Opening**
**Solution**:
```bash
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```
If False, check camera permissions or try different camera index.

### **Issue: Low Recognition Accuracy**
**Solution**:
- Use better quality images (face visible, no occlusion)
- Add more images per person (5-10 images)
- Ensure good lighting
- Different angles/expressions

---

## 📚 Model Information

### **YOLO (YOLOv8n-face)**
- Purpose: Face detection
- Input: Any resolution (auto-scaled)
- Output: Bounding boxes with confidence
- Speed: 30-50 FPS

### **InsightFace (buffalo_l)**
- **SCRFD** (Face Detection): 
  - Multi-scale face detection
  - Resolution: Any (auto-resize)
  
- **ArcFace** (Face Recognition):
  - Embedding dim: 512
  - Pre-trained on 5.8M+ faces
  - Similarity: Cosine distance

---

## 📦 Creating New Embeddings

```bash
# 1. Add new person images
mkdir data/known_faces/new_person
# Copy images to this folder

# 2. Rebuild embeddings
python src/recognition/build_embeddings.py

# 3. Run recognition again
python src/recognition/recognize_camera.py
```

The system automatically detects new images and updates the database.

---

## 🔍 Database Structure

**Face Embeddings File**: `embeddings/face_embeddings.pkl`

Structure:
```python
{
    "person_name_1": [
        embedding_array_1,  # 512-dim numpy array
        embedding_array_2,
        ...
    ],
    "person_name_2": [
        embedding_array_1,
        ...
    ]
}
```

---

## 📞 Support & Notes

- **Python Version**: 3.11.9 (recommended)
- **OS**: Windows/Linux/macOS
- **GPU Support**: CUDA 11.8+ (optional, CPU works fine)
- **Internet**: Required for first-time model download

---

## 🎓 Learning Resources

- **InsightFace**: https://github.com/deepinsight/insightface
- **YOLO**: https://github.com/ultralytics/yolov8
- **OpenCV**: https://docs.opencv.org/

---

## 📄 License & Credits

- InsightFace: BSD 2-Clause
- YOLO: AGPL-3.0
- OpenCV: Apache 2.0

---

**Last Updated**: January 3, 2026  
**Project Status**: Phase-2 Complete ✅  
**Next Phase**: API Integration (Phase-3)

