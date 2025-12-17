import os
from ultralytics import YOLO
import yaml

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(base_dir, 'data.yaml')
    
    print("Loading YOLOv8 Nano model for fast CPU training...")
    model = YOLO('yolov8n.pt') 

    print("Starting FAST training...")
    try:
        results = model.train(
            data=yaml_file,
            epochs=30,               
            imgsz=416,               
            project='runs/detect',
            name='train_fast',
            exist_ok=True,
            patience=5,              
            batch=8,                 
            
            hsv_h=0.01,
            hsv_s=0.1,
            hsv_v=0.1,
            degrees=0.0,
            translate=0.1,
            scale=0.1,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.0,              
            mixup=0.0,               
            
            optimizer='AdamW',       
            plots=True,
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    main()
