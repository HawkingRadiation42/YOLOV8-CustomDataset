from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("runs/detect/train3/weights/best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")

# from PIL
# im1 = Image.open("datasets/test/images/image_0051_jpg.rf.49c71c882683e210a148dd1f87ce6096.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
# im2 = cv2.imread("datasets/test/images/image_0122_jpg.rf.f8708c5037466fd5e4f9d83356ff5a18.jpg")
# results = model.predict(source=im2, save=True)#, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
# results = model.predict(source=[im1, im2], save=True)
results = model.predict(source="datasets/test/images", save=True) # Display preds. Accepts all YOLO predict arguments