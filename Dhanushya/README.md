**Caterpillar Autonomy Challenge — Object Detection (Colab Training Pipeline)**

This notebook provides a complete workflow to train, validate, export, and deploy a custom object detection model for the Caterpillar Autonomy Challenge arena using the YOLO11 architecture.
It is designed to work entirely inside Google Colab, with seamless integration to Google Drive, custom datasets, and Jetson deployment.

**Features of This Notebook**

✔ Mount Google Drive to store datasets and model outputs
✔ Automatically generate data.yaml from class list
✔ Organize dataset into YOLO format (train/val folders)
✔ Train YOLO11s with custom hyperparameters
✔ Save trained weights in Google Drive
✔ Export model to TFLite, ONNX, and TensorRT (engine) formats
✔ Ready-to-run Jetson Orin Nano inference script
✔ GPU-accelerated training with Colab T4/A100

📁**Dataset Requirements**

Dataset must contain:

images/        # raw images
labels/        # YOLO .txt files
classes.txt    # list of class names (one per line)

During notebook execution, it will be automatically organized into:

/content/data/
 ├── train/images
 ├── train/labels
 ├── val/images
 ├── val/labels
 └── data.yaml

**Install Requirements (Ultralytics)**
!pip install ultralytics

** Model Training (YOLO11s)**

Training is done with:

!yolo detect train \
    data=/content/data.yaml \
    model=yolo11s.pt \
    imgsz=640 \
    epochs=60


Output:
runs/detect/train/weights/best.pt

This model is saved to Drive as well.

**Export Options**

After training, the notebook converts the model into multiple deployment formats:

1. TFLite
!yolo export model=best.pt format=tflite

2. ONNX
!yolo export model=best.pt format=onnx

3. TensorRT (Jetson Orin Nano)
!yolo export model=best.pt format=engine


All exported models are stored in:
runs/detect/train/weights/

**Test Model**

!yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True

import glob
from IPython.display import Image, display
for image_path in glob.glob(f'/content/runs/detect/predict/*.jpg')[:10]:
  display(Image(filename=image_path, height=400))
  print('\n')

  You can also run the model on video files or other images images by uploading them to this notebook and using the above !yolo detect predict command, where source points to the location of the video file, image, or folder of images. The results will be saved in runs/detect/predict.

  **Deploy Model**
  First, zip and download the trained model 

  Use Anaconda Prompt to run Ultralytics models

1, Set up virtual environment

Once it's installed, run Anaconda Prompt from the Start Bar. (If you're on macOS or Linux, just open a command terminal).
Issue the following commands to create a new Python environment and activate it:

conda create --name yolo-env1 python=3.12 -y
conda activate yolo-env1

Install Ultralytics (which also installs import libraries like OpenCV-Python, Numpy, and PyTorch) by issuing the following command:
pip install ultralytics

If you have an NVIDIA GPU, you can install the GPU-enabled version of PyTorch by issuing the following command:
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

2. Extract downloaded model Take the my_model.zip file you downloaded in Step 7.1 and unzip it to a folder on your PC. In the Anaconda Prompt terminal, move into the unzipped folder using:
cd path/to/folder

4. Download and run yolo_detect.py
Download the yolo_detect.py script into the my_model folder using:

curl -o yolo_detect.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/yolo_detect.py
Alright! We're ready to run the script. To run inference with a yolov8s model on a USB camera, issue:

python yolo_detect.py --model my_model.pt --source usb0 
