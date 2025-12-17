import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Rover AI Vision (Light)", layout="wide")

st.title("🚜 Caterpillar-AI Autonomous Rover Vision (Light)")
st.markdown("""
**Lightweight Autonomous Navigation System**
Uses **YOLOv8 TFLite** for fast object detection and a **Virtual Depth Map** for path planning.
""")

st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
num_sectors = st.sidebar.slider("Navigation Sectors", 10, 60, 32, 2)
smoothing_factor = st.sidebar.slider("Stability (Smoothing)", 0.0, 0.95, 0.6, 0.05)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for the quantized tflite model first
    model_path = os.path.join(base_dir, 'best_int8.tflite')
    
    # Fallback to checking if it's just best.tflite or similar
    if not os.path.exists(model_path):
        # Try finding any .tflite or .onnx file
        for file in os.listdir(base_dir):
            if file.endswith(".tflite") or file.endswith(".onnx"):
                model_path = os.path.join(base_dir, file)
                break
    
    print(f"Loading Light model from: {model_path}")
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

class RoverNavigator:
    def __init__(self, width, height, sectors=32, smoothing=0.6):
        self.width = width
        self.height = height
        self.sectors = sectors
        self.sector_width = width // sectors
        self.smoothing = smoothing
        
        # State variables for smoothing
        self.smoothed_depths = np.full(sectors, height, dtype=float)
        self.target_sector = sectors // 2
        
    def update(self, detections):
        # 1. Calculate Raw Depth Map
        raw_depths = np.full(self.sectors, self.height, dtype=int)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Map bounding box to sectors
            start = max(0, x1 // self.sector_width)
            end = min(self.sectors - 1, x2 // self.sector_width)
            
            # Distance metric: Distance from bottom of screen to bottom of box
            dist = self.height - y2
            if dist < 0: dist = 0
            
            # Update sectors with minimum distance (closest obstacle)
            for s in range(start, end + 1):
                if dist < raw_depths[s]:
                    raw_depths[s] = dist
                    
        # 2. Temporal Smoothing (Exponential Moving Average)
        self.smoothed_depths = (self.smoothing * self.smoothed_depths) + ((1 - self.smoothing) * raw_depths)
        
        # 3. Find Best Path (Convolution/Sliding Window)
        window_size = max(3, self.sectors // 8) 
        best_score = -1
        best_idx = self.sectors // 2
        
        for i in range(self.sectors - window_size + 1):
            window_score = np.mean(self.smoothed_depths[i : i+window_size])
            center_offset = abs((i + window_size/2) - (self.sectors/2))
            penalty = center_offset * (self.height * 0.01)
            final_score = window_score - penalty
            
            if final_score > best_score:
                best_score = final_score
                best_idx = i + window_size // 2
                
        self.target_sector = int((self.smoothing * self.target_sector) + ((1 - self.smoothing) * best_idx))
        
        return self.smoothed_depths, self.target_sector, self.sector_width

    def get_command(self):
        center = self.sectors // 2
        depth_at_target = self.smoothed_depths[self.target_sector]
        
        safe_distance = self.height * 0.2
        speed = min(1.0, max(0.0, (depth_at_target - safe_distance) / (self.height * 0.6)))
        
        if speed < 0.1:
            return "STOP 🛑", (0, 0, 255), speed
            
        deadzone = self.sectors * 0.1
        
        if abs(self.target_sector - center) < deadzone:
            return "FORWARD ⬆️", (0, 255, 0), speed
        elif self.target_sector < center:
            return "LEFT ⬅️", (255, 255, 0), speed
        else:
            return "RIGHT ➡️", (255, 255, 0), speed

# --- Main App Logic ---

input_mode = st.sidebar.radio("Select Input Mode", ["Video", "Image"])

if input_mode == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        st_metrics = st.empty()
        
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            navigator = RoverNavigator(w, h, sectors=num_sectors, smoothing=smoothing_factor)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame = cv2.resize(frame, (640, 480))
            
            # 1. Object Detection (TFLite)
            if model:
                results = model(frame, verbose=False, conf=confidence_threshold)
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({'bbox': (int(x1), int(y1), int(x2), int(y2))})
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            else:
                detections = []

            # 2. Navigation Update
            depths, target_sec, nav_width = navigator.update(detections)
            cmd, color, speed = navigator.get_command()
            
            # 3. Visualization
            overlay = frame.copy()
            for i, d in enumerate(depths):
                h_bar = int((d / navigator.height) * 100)
                c_val = int(255 * (d / navigator.height))
                cv2.rectangle(overlay, 
                             (i*nav_width, navigator.height - h_bar),
                             ((i+1)*nav_width, navigator.height),
                             (0, c_val, 255 - c_val), -1)
            
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            start_pt = (navigator.width // 2, navigator.height)
            end_pt = (int(target_sec * nav_width + nav_width/2), navigator.height - 150)
            cv2.arrowedLine(frame, start_pt, end_pt, (255, 0, 255), 4)
            
            cv2.putText(frame, f"{cmd} | Speed: {int(speed*100)}%", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame, use_container_width=True)
            
            with st_metrics.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Command", cmd)
                c2.metric("Speed", f"{int(speed*100)}%")
                c3.metric("Safety Score", f"{int(np.mean(depths))}")

        cap.release()

elif input_mode == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]
        
        navigator = RoverNavigator(w, h, sectors=num_sectors, smoothing=0.0)
        
        if model:
            results = model(frame, verbose=False, conf=confidence_threshold)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({'bbox': (int(x1), int(y1), int(x2), int(y2))})
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        else:
            detections = []
        
        depths, target_sec, nav_width = navigator.update(detections)
        cmd, color, speed = navigator.get_command()
        
        overlay = frame.copy()
        for i, d in enumerate(depths):
            h_bar = int((d / navigator.height) * 100)
            c_val = int(255 * (d / navigator.height))
            cv2.rectangle(overlay, 
                         (i*nav_width, navigator.height - h_bar),
                         ((i+1)*nav_width, navigator.height),
                         (0, c_val, 255 - c_val), -1)
        
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        start_pt = (navigator.width // 2, navigator.height)
        end_pt = (int(target_sec * nav_width + nav_width/2), navigator.height - 150)
        cv2.arrowedLine(frame, start_pt, end_pt, (255, 0, 255), 4)
        
        cv2.putText(frame, f"{cmd} | Speed: {int(speed*100)}%", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Analyzed Image", use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Command", cmd)
        c2.metric("Speed", f"{int(speed*100)}%")
        c3.metric("Safety Score", f"{int(np.mean(depths))}")
