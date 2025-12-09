import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the Object Detector with a YOLO model.
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

    def detect(self, frame):
        """
        Detect objects in the given frame.
        Returns the frame with annotations and a list of detections.
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                label = self.model.names[cls]

                detections.append({
                    'label': label,
                    'confidence': float(conf),
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })

        # Plot results on the frame
        annotated_frame = results[0].plot()
        return annotated_frame, detections

if __name__ == "__main__":
    # Simple test
    detector = ObjectDetector()
    # Create a dummy image or capture from webcam for testing
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()
    
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, dets = detector.detect(frame)
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
