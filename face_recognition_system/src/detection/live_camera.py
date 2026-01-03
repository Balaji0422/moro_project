import cv2
import time
from pathlib import Path

from yolo_detector import YOLOFaceDetector


# ---------------- CONFIG ----------------
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "yolo" / "yolov8n.pt"
WINDOW_NAME = "YOLOv8 Detection - LIVE"
# ----------------------------------------


def main():
    print("\n🔄 Loading YOLO model...")
    detector = YOLOFaceDetector(str(MODEL_PATH))
    print("✓ Model loaded")

    print("\n📷 Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW is IMPORTANT on Windows

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✓ Camera opened")

    # ---- FORCE WINDOW CREATION (WINDOWS FIX) ----
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED
    )
    cv2.resizeWindow(WINDOW_NAME, 900, 600)

    print("\n🎥 LIVE DETECTION STARTED")
    print("Press Q or ESC to quit\n")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        frame_count += 1

        # ---- DETECTION ----
        faces = detector.detect(frame)

        for (x1, y1, x2, y2, conf) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Det {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ---- FPS ----
        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        cv2.putText(
            frame,
            f"FPS: {int(fps)} | Faces: {len(faces)} | Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2
        )

        # ---- SHOW FRAME (NO TRY/EXCEPT) ----
        cv2.imshow(WINDOW_NAME, frame)

        # ---- FORCE WINDOW TO FRONT ----
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_TOPMOST,
            1
        )

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            print("\n🛑 Stopping...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Camera closed")


if __name__ == "__main__":
    main()
