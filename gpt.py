import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

def load_model(path):
    model = YOLO(path)
    return model

def detect_license_plate(image_path, model):
    img = cv2.imread(image_path)

    results = model(img)

    roi_folder = "ROI"
    if not os.path.exists(roi_folder):
        os.makedirs(roi_folder)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            r = box.xyxy[0].astype(int)
            crop = img[r[1]:r[3], r[0]:r[2]]

            # Generate a unique filename using the index i
            filename = f"roi_{i}_{os.path.basename(image_path)}"
            roi_path = os.path.join(roi_folder, filename)
            cv2.imwrite(roi_path, crop)
            text = ocr_model.ocr(roi_path)
            print(f"License Plate Text: {text}")


if __name__ == "__main__":
    model_path = 'runs/detect/train3/weights/best.pt'
    image_folder = 'datasets/test/images'

    model = load_model(model_path)
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            detect_license_plate(image_path, model)
