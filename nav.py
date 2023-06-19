import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

def load_model(path):
    model = YOLO(path)
    return model

def detect_license_plate(image_path, model, ocr_model):
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
            # filename = "roi.jpg"
            roi_path = os.path.join(roi_folder, filename)
            cv2.imwrite(roi_path, crop)

            print(f"Saved cropped image: {roi_path}")
            # Initialize list to hold all accuracies
            all_accuracies = []

            for filename in os.listdir(roi_folder):
                roi_image_path = os.path.join(roi_folder, filename)
                results = ocr_model.ocr(roi_image_path)
                texts_and_accuracies = [(result[1][0], result[1][1]) for sublist in results for result in sublist]

                # Get the list of texts
                texts = [text for text, accuracy in texts_and_accuracies]
                combined_text = ', '.join(texts)
                no_special_characters_text = combined_text.replace(', ', '').replace(' ', '')

                # Calculate the average accuracy
                accuracies = [accuracy for text, accuracy in texts_and_accuracies]
                if len(accuracies) > 0:
                    average_accuracy = sum(accuracies) / len(accuracies)
                    print(
                        f"License Plate Text for {filename}: {no_special_characters_text} with average accuracy {average_accuracy}")
                    all_accuracies.extend(accuracies)  # Add the accuracies to the list of all accuracies
                else:
                    print(f"License Plate Text for {filename}: {no_special_characters_text} with no detected texts")

            # Calculate and print the final average accuracy over all images
            if len(all_accuracies) > 0:
                final_average_accuracy = sum(all_accuracies) / len(all_accuracies)
                print(f"\nFinal average accuracy over all images: {final_average_accuracy}")
            else:
                print("\nNo text was detected in any image.")


if __name__ == "__main__":
    # Perform OCR on the saved images

    model_path = 'runs/detect/train3/weights/best.pt'
    image_folder = 'datasets/test/images'

    model = load_model(model_path)
    ocr_model = PaddleOCR(lang='en')

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image_path = os.path.join(image_folder, filename)
            detect_license_plate(image_path, model, ocr_model)