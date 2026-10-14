import os
import urllib.request
import cv2

MODEL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
PHOTO = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
CONFIDENCE_THRESHOLD = 0.2

script_dir = os.path.dirname(os.path.abspath(__file__))
files = (
    ("deploy.prototxt", MODEL + "deploy.prototxt"),
    ("mobilenet_iter_73000.caffemodel", MODEL + "mobilenet_iter_73000.caffemodel"),
    ("bus.jpg", PHOTO),
)

for name, url in files:
    path = os.path.join(script_dir, name)
    if not os.path.exists(path):
        print("Downloading", name)
        urllib.request.urlretrieve(url, path)

image_path = os.path.join(script_dir, "bus.jpg")
network = cv2.dnn.readNetFromCaffe(
    os.path.join(script_dir, "deploy.prototxt"),
    os.path.join(script_dir, "mobilenet_iter_73000.caffemodel"),
)
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=0.007843,
    size=(300, 300),
    mean=(127.5, 127.5, 127.5),
)
network.setInput(blob)
detections = network.forward()

height, width = image.shape[:2]
found = 0
for detection in detections[0, 0]:
    confidence = float(detection[2])
    if confidence < CONFIDENCE_THRESHOLD:
        continue

    class_id = int(detection[1])
    x1, y1, x2, y2 = (
        int(detection[3] * width),
        int(detection[4] * height),
        int(detection[5] * width),
        int(detection[6] * height),
    )
    label = f"{CLASSES[class_id]}: {confidence:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        label,
        (max(x1, 15), max(y1, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    print(label)
    found += 1

output_path = os.path.join(script_dir, "bus_detected.jpg")
if not cv2.imwrite(output_path, image):
    raise OSError(f"Could not save detection result to: {output_path}")
print(f"Detected {found} object(s). Result saved to {output_path}")