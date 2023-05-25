import os
import re
import cv2
import torch
from PIL import Image
import pytesseract
from ultralytics import YOLO

def load_model(path):
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
    model = YOLO(path)
    return model

def detect_license_plate(image_path, model):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Convert to tensor

    # Inference
    results = model(img)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            r = box.xyxy[0].astype(int)
            crop = img[r[1]:r[3], r[0]:r[2]]      
            roi_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
            print(f"License Plate Text: {text}")


if __name__ == "__main__":
    model_path = 'runs/detect/train3/weights/best.pt'
    # image_folder = 'datasets/test/images'  # replace with your folder path
    image_folder = 'runs/detect/predict2'  # replace with your folder path

    model = load_model(model_path)
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # check for image file extensions
            image_path = os.path.join(image_folder, filename)
            detect_license_plate(image_path, model)
