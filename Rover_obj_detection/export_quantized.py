from ultralytics import YOLO
import os

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'runs', 'detect', 'train_fast', 'weights', 'best.pt')

print(f"Loading model from: {model_path}")

# Load the model
model = YOLO(model_path)

# Export the model to TFLite format with INT8 quantization
print("Starting export to TFLite (INT8 Quantization)...")
# int8=True enables quantization, which reduces size and improves speed on supported hardware
# data=... is usually needed for int8 to calibrate, but ultralytics handles it with default coco128 if not specified, 
# or uses the data from the model if available.
model.export(format='tflite', int8=True)

print("Export complete!")
