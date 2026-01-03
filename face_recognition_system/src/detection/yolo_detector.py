"""
YOLO Face Detection Module
Handles face detection using YOLOv8 pretrained model
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO


class YOLODetector:
    """YOLOv8-based face detector"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.45):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model (.pt file)
                       If None, uses default 'yolov8n-face.pt'
            conf_threshold: Confidence threshold for detections (0.0-1.0)
        """
        if model_path is None:
            # Default to project model location - using face-specific model
            model_path = str(Path(__file__).parent.parent.parent / 'models' / 'yolo' / 'yolov8n-face.pt')
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✓ Model loaded: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect(self, image: np.ndarray) -> dict:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format, numpy array)
        
        Returns:
            dict with keys:
                - 'detections': List of detections
                - 'image_shape': (height, width, channels)
                - 'count': Number of faces detected
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image input")
        
        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        # Extract detections
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detection = {
                    'bbox': (x1, y1, x2, y2),  # (x1, y1, x2, y2)
                    'confidence': conf,
                    'class': cls,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                detections.append(detection)
        
        return {
            'detections': detections,
            'image_shape': image.shape,
            'count': len(detections)
        }
    
    def detect_file(self, image_path: str) -> dict:
        """
        Detect faces in image file
        
        Args:
            image_path: Path to image file
        
        Returns:
            dict with detection results and image data
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        return self.detect(image)
    
    def draw_detections(self, image: np.ndarray, detections: List[dict], 
                       color: Tuple[int, int, int] = (0, 255, 0), 
                       thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image (BGR)
            detections: List of detection dicts
            color: Box color in BGR format
            thickness: Line thickness
        
        Returns:
            Image with drawn detections
        """
        image_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence score
            text = f"Face: {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            # Background for text
            cv2.rectangle(image_copy, 
                         (x1, y1 - text_size[1] - 5),
                         (x1 + text_size[0] + 5, y1),
                         color, -1)
            
            # Text
            cv2.putText(image_copy, text, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image_copy
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set detection confidence threshold
        
        Args:
            threshold: Confidence threshold (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.conf_threshold = threshold
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'confidence_threshold': self.conf_threshold,
            'model_type': 'YOLOv8',
            'model_size': 'nano' if 'yolov8n' in self.model_path else 'unknown'
        }


class YOLOFaceDetector:
    """Simplified YOLO face detector with tuple output"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.45):
        """Initialize detector"""
        self.detector = YOLODetector(model_path, conf_threshold)
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces and return as list of tuples
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        result = self.detector.detect(frame)
        detections = []
        
        for det in result['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            detections.append((x1, y1, x2, y2, conf))
        
        return detections


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = YOLODetector()
    
    # Test with webcam
    print("Testing with webcam... Press 'q' to quit")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            result = detector.detect(frame)
            
            # Draw detections
            frame_with_boxes = detector.draw_detections(frame, result['detections'])
            
            # Display
            cv2.putText(frame_with_boxes, 
                       f"Faces detected: {result['count']}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            cv2.imshow('YOLO Face Detection', frame_with_boxes)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
