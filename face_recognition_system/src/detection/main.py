import cv2
import time
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from yolo_detector import YOLOFaceDetector

MODEL_PATH = str(Path(__file__).parent.parent.parent / "models" / "yolo" / "yolov8n.pt")

def main():
    print(f"\n{'='*60}")
    print("🎥 FACE DETECTION SYSTEM - OPTIMIZED")
    print(f"{'='*60}\n")
    
    print("🔄 Loading model...")
    detector = YOLOFaceDetector(MODEL_PATH, conf_threshold=0.5)
    print("✓ Model loaded\n")

    print("📷 Opening camera...\n")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return 1
    
    # Optimize camera settings for speed
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
    
    # Get camera properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Camera opened!")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}\n")
    
    # Setup video output
    output_dir = Path(__file__).parent.parent.parent / 'outputs' / 'detections'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = output_dir / f"detection_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    print(f"📹 Recording to: {video_path}\n")
    print("="*60)
    print("🎬 LIVE DETECTION STARTED")
    print("="*60 + "\n")

    prev_time = time.time()
    frame_count = 0
    total_faces = 0
    detect_skip = 2  # Process every 2nd frame for speed
    detection_frame_count = 0
    last_faces = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Only detect every N frames to speed up
            if frame_count % detect_skip == 0:
                detection_frame_count += 1
                faces = detector.detect(frame)
                last_faces = faces
                total_faces += len(faces)
            else:
                # Use last detection result for smooth display
                faces = last_faces
            
            # Draw detections
            for (x1, y1, x2, y2, conf) in faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
            
            # FPS calculation
            curr_time = time.time()
            fps_current = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Add info (minimal text for speed)
            cv2.putText(
                frame,
                f"FPS: {int(fps_current)} | Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                1
            )
            
            # Save to video
            video_writer.write(frame)
            
            # Display
            try:
                cv2.imshow("YOLOv8 Face Detection - LIVE", frame)
            except cv2.error:
                if frame_count % 30 == 0:
                    print(f"  ✓ Frame {frame_count} | Faces: {len(faces)}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n🛑 Stopping...")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 User interrupted")
    
    finally:
        cap.release()
        video_writer.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    print("\n" + "="*60)
    print("✅ DETECTION COMPLETE")
    print("="*60)
    print(f"  Total frames: {frame_count}")
    print(f"  Detection runs: {detection_frame_count}")
    print(f"  Total faces: {total_faces}")
    if detection_frame_count > 0:
        print(f"  Avg faces: {total_faces/detection_frame_count:.2f}")
    print(f"\n  📁 Video: {video_path}")
    print("="*60 + "\n")
    
    return 0

if __name__ == "__main__":
    exit(main())
