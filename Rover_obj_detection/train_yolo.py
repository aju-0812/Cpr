import os
import shutil
import random
from ultralytics import YOLO
import yaml

def prepare_dataset(base_dir):
    data_dir = os.path.join(base_dir, 'data')
    source_images = os.path.join(base_dir, 'images')
    source_labels = os.path.join(base_dir, 'labels')

    if not os.path.exists(source_images) or not os.path.exists(source_labels):
        if os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, 'train', 'images')):
            print("Dataset appears to be already organized in 'data' folder.")
            return True
        print("Error: 'images' and 'labels' folders not found in current directory, and 'data' folder is missing or incomplete.")
        return False

    print("Organizing dataset...")
    
    for split in ['train', 'validation']:
        os.makedirs(os.path.join(data_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, split, 'labels'), exist_ok=True)

    image_files = [f for f in os.listdir(source_images) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.shuffle(image_files)

    split_ratio = 0.8 
    split_index = int(len(image_files) * split_ratio)
    
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    def move_files(files, split):
        for img_file in files:
            src_img = os.path.join(source_images, img_file)
            dst_img = os.path.join(data_dir, split, 'images', img_file)
            shutil.move(src_img, dst_img)

            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(source_labels, label_file)
            dst_label = os.path.join(data_dir, split, 'labels', label_file)
            
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
            else:
                print(f"Warning: Label file not found for {img_file}")

    move_files(train_files, 'train')
    move_files(val_files, 'validation')
    
    try:
        os.rmdir(source_images)
        os.rmdir(source_labels)
    except:
        pass 

    print(f"Dataset organized: {len(train_files)} train, {len(val_files)} validation.")
    return True

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    if not os.path.exists(path_to_classes_txt):
        print(f'Error: classes.txt file not found at {path_to_classes_txt}')
        return False

    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    number_of_classes = len(classes)
    
    data = {
        'path': os.path.abspath('data'), 
        'train': 'train/images',
        'val': 'validation/images',
        'nc': number_of_classes,
        'names': classes
    }

    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f'Created config file at {path_to_data_yaml}')
    return True

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not prepare_dataset(base_dir):
        return

    data_dir = os.path.join(base_dir, 'data')
    classes_file = os.path.join(base_dir, 'classes.txt')
    if not os.path.exists(classes_file):
         classes_file = os.path.join(data_dir, 'classes.txt')
    
    if not os.path.exists(classes_file):
        print("Error: classes.txt not found.")
        return

    yaml_file = os.path.join(base_dir, 'data.yaml')
    
    if not create_data_yaml(classes_file, yaml_file):
        return

    try:
        model = YOLO('yolo11s.pt') 
    except Exception as e:
        print(f"Could not load yolo11s.pt, falling back to yolov8n.pt. Error: {e}")
        model = YOLO('yolov8n.pt')

    print("Starting training with optimized settings...")
    try:
        results = model.train(
            data=yaml_file,
            epochs=100,              
            imgsz=640,
            project='runs/detect',
            name='train',
            exist_ok=True,
            patience=20,             
            batch=16,                
            
            hsv_h=0.015,             
            hsv_s=0.7,               
            hsv_v=0.4,               
            degrees=10.0,            
            translate=0.1,           
            scale=0.5,               
            shear=2.0,               
            perspective=0.0,         
            flipud=0.0,              
            fliplr=0.5,              
            mosaic=1.0,              
            mixup=0.1,               
            copy_paste=0.1,          
            
            optimizer='auto',        
            lr0=0.01,                
            lrf=0.01,                
            momentum=0.937,          
            weight_decay=0.0005,     
            warmup_epochs=3.0,       
            warmup_momentum=0.8,     
            warmup_bias_lr=0.1,      
            box=7.5,                 
            cls=0.5,                 
            dfl=1.5,                 
            pose=12.0,               
            kobj=1.0,                
            label_smoothing=0.0,     
            nbs=64,                  
            overlap_mask=True,       
            mask_ratio=4,            
            dropout=0.0,             
            val=True,                
            plots=True,              
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    main()
