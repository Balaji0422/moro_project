from ultralytics import YOLO
import os

# Change to models/yolo directory
os.chdir(r'c:\infiposts_project\Moro_project_python_1\face_recognition_system\models\yolo')

print("🔄 Downloading YOLOv8n model...")
try:
    model = YOLO('yolov8n.pt')
    print("✓ YOLOv8n model downloaded successfully!")
    print(f"  Location: {os.path.abspath('yolov8n.pt')}")
except Exception as e:
    print(f"Error downloading: {e}")

print("\n📌 For Face Detection specifically:")
print("   Option 1: pip install ultralytics-yoloface")
print("   Option 2: Download pretrained face weights from:")
print("   https://github.com/akanametov/yolov8-face")
