from ultralytics import YOLO
import os

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'runs', 'detect', 'train_fast', 'weights', 'best.pt')

print(f"Loading model from: {model_path}")

# Load the model
model = YOLO(model_path)

# Export the model to TFLite format
print("Starting export to TFLite...")
# int8 quantization is often good for speed/size, but let's stick to standard float32 or float16 first for compatibility
# We can use 'int8' argument if needed, but default is usually float32/16
model.export(format='tflite')

print("Export complete!")
