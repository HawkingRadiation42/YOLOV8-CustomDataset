from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import pytesseract
import torch
# from yolov5.models.yolo import Model
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
import os 

def load_model(path):
    # ckpt = torch.load(path, map_location="cpu")
    # model_yaml = ckpt['model'].yaml
    # model_yaml['anchors'] = 3 # add 'anchors' key
    model = YOLO(path)  # load a pretrained model (recommended for training)
    # metrics = model.val()  # evaluate model performance on the validation set
    return model

def detect_license_plate(image_path, model):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Convert to tensor

    # Inference
    results = model(img)  # predict on an image
    # results = model(img_tensor)
    # Extract bounding boxes
    boxes = results.xyxy[0].cpu().detach().numpy()

    # Filter boxes with label corresponding to 'license plate' if you've trained for other objects too
    license_plates = [box for box in boxes if box[5] == 0]
    ocr_model = PaddleOCR(lang='en')
    # Assuming 'license_plates' now contains boxes around license plates, extract these regions
    for i, (x1, y1, x2, y2, conf, label) in enumerate(license_plates):
        # Crop the license plate out of the original image
        license_plate_img = img[int(y1):int(y2), int(x1):int(x2)]
        
        # Convert image to grayscale
        # gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

        # Use Tesseract to perform OCR on the extracted license plate
        result = ocr_model.ocr(license_plate_img)
        for res in result:
            print(f"License Plate Text {i + 1}: {res[1][0]}")


if __name__ == "__main__":
    model_path = 'runs/detect/train3/weights/best.pt'
    image_path = 'datasets/test/images/image_0009_jpg.rf.69c3f5b1dc1af7c0741b11ea54dd49c8.jpg'
    model = load_model(model_path)
    results = model(image_path)  # predict on an image
    # plt.imshow(results)
    print("hello")
    print(results)
    # cv2.imwrite('plot.png', results)
    # detect_license_plate(image_path, model)