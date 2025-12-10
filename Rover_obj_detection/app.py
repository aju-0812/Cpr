import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Rover AI Vision", layout="wide")

st.title("🚜 Caterpillar-AI Autonomous Rover Vision")
st.markdown("""
Upload a video to detect obstacles (Boulders, Craters, etc.) and get real-time navigation commands.
""")

st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    fast_model_path = os.path.join(base_dir, 'runs', 'detect', 'train_fast', 'weights', 'best.pt')
    normal_model_path = os.path.join(base_dir, 'runs', 'detect', 'train', 'weights', 'best.pt')
    
    if os.path.exists(fast_model_path):
        return YOLO(fast_model_path)
    elif os.path.exists(normal_model_path):
        return YOLO(normal_model_path)
    else:
        st.warning("Trained model not found. Using default YOLOv8n model.")
        return YOLO('yolov8n.pt')

model = load_model()

def get_navigation_command(frame_width, detections):
    left_limit = frame_width // 3
    right_limit = 2 * (frame_width // 3)
    
    zones = {'left': 0, 'center': 0, 'right': 0}
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) // 2
        
        if cx < left_limit:
            zones['left'] += 1
        elif cx < right_limit:
            zones['center'] += 1
        else:
            zones['right'] += 1
            
    command = "STOP"
    color = (0, 0, 255) 
    
    if zones['center'] == 0:
        command = "GO STRAIGHT"
        color = (0, 255, 0) 
    elif zones['left'] == 0:
        command = "TURN LEFT"
        color = (255, 255, 0) 
    elif zones['right'] == 0:
        command = "TURN RIGHT"
        color = (255, 255, 0) 
    else:
        if zones['left'] <= zones['right']:
             command = "TURN LEFT (Caution)"
             color = (255, 165, 0) 
        else:
             command = "TURN RIGHT (Caution)"
             color = (255, 165, 0) 
             
    return command, color, zones

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    st_info = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]
        
        results = model(frame, verbose=False, conf=confidence_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                label = model.names[cls]
                
                detections.append({
                    'label': label,
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        command, color, zones = get_navigation_command(width, detections)
        
        cv2.line(frame, (width//3, 0), (width//3, height), (200, 200, 200), 1)
        cv2.line(frame, (2*width//3, 0), (2*width//3, height), (200, 200, 200), 1)
        
        cv2.putText(frame, f"CMD: {command}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame, channels="RGB", use_container_width=True)
        
        with st_info.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Left Zone Obstacles", zones['left'])
            c2.metric("Center Zone Obstacles", zones['center'])
            c3.metric("Right Zone Obstacles", zones['right'])

    cap.release()
