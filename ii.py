import os
import urllib.request
import cv2

MODEL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
CONFIDENCE_THRESHOLD = 0.2

script_dir = os.path.dirname(os.path.abspath(__file__))
model_files = (
    ("deploy.prototxt", MODEL + "deploy.prototxt"),
    ("mobilenet_iter_73000.caffemodel", MODEL + "mobilenet_iter_73000.caffemodel"),
)

for name, url in model_files:
    path = os.path.join(script_dir, name)
    if not os.path.exists(path):
        print("Downloading", name)
        urllib.request.urlretrieve(url, path)

network = cv2.dnn.readNetFromCaffe(
    os.path.join(script_dir, "deploy.prototxt"),
    os.path.join(script_dir, "mobilenet_iter_73000.caffemodel"),
)

def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
    )
    network.setInput(blob)
    detections = network.forward()

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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (max(x1, 15), max(y1, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        found += 1

    return found


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam. Check that a camera is connected.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            found = detect_objects(frame)
            cv2.putText(
                frame,
                f"Objects: {found}  |  Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Live Webcam - Object Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
