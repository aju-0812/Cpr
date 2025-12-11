import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                label = self.model.names[cls]

                detections.append({
                    'label': label,
                    'confidence': float(conf),
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })

        annotated_frame = results[0].plot()
        return annotated_frame, detections

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    fast_model_path = os.path.join(base_dir, 'runs', 'detect', 'train_fast', 'weights', 'best.pt')
    normal_model_path = os.path.join(base_dir, 'runs', 'detect', 'train', 'weights', 'best.pt')
    
    if os.path.exists(fast_model_path):
        print(f"Found FAST trained model at {fast_model_path}")
        model_to_use = fast_model_path
    elif os.path.exists(normal_model_path):
        print(f"Found trained model at {normal_model_path}")
        model_to_use = normal_model_path
    else:
        print("Trained model not found. Using default yolov8n.pt")
        model_to_use = 'yolov8n.pt'

    detector = ObjectDetector(model_path=model_to_use)
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
        cv2.imshow('YOLO Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()