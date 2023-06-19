from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

def load_model(path):
    model = YOLO(path)
    return model

def modify_string(ocr_results):
    modified_strings = []
    confidences = []

    for sublist in ocr_results:
        for result in sublist:
            if len(result) > 1 and result[1]:
                string = result[1][0]
                confidence = result[1][1]
                modified_string = string.replace('-', '').replace('IND', '').replace('.', '').replace(" ",'')
                modified_strings.append(modified_string)
                confidences.append(confidence)

    return modified_strings, confidences

def detect_license_plate(image_path, model, ocr_model):
    img = cv2.imread(image_path)

    results = model(img)

    roi_folder = "ROI"
    if not os.path.exists(roi_folder):
        os.makedirs(roi_folder)

    detected_texts = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            r = box.xyxy[0].astype(int)
            crop = img[r[1]:r[3], r[0]:r[2]]

            filename = f"roi_{i}.jpg"
            roi_path = os.path.join(roi_folder, filename)
            cv2.imwrite(roi_path, crop)

            logging.info(f"Saved cropped image: {roi_path}")

            text = ocr_model.ocr(roi_path)
            text, _ = modify_string(text)
            text_string = ''.join(text)
            text_string = text_string[:10]

            detected_texts.append(text_string)

            logging.info(f"License Plate Text for {filename}: {text_string}")

    return detected_texts

@app.route('/detect-text', methods=['POST'])
def handle_text_detection():
    try:
        image_file = request.files['image']
        image_path = 'uploads/temp.jpg'
        image_file.save(image_path)

        model_path = 'runs/detect/train3/weights/best.pt'
        model = load_model(model_path)
        ocr_model = PaddleOCR(lang='en')

        detected_texts = detect_license_plate(image_path, model, ocr_model)

        response = {'detectedTexts': detected_texts}
        logging.info(f"Detected texts: {detected_texts}")

        return jsonify(response)
    except Exception as e:
        logging.error(str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '_main_':
    app.run(debug=True)