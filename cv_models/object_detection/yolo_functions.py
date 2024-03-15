from ultralytics import YOLO
from ultralytics import YOLOWorld
import cvzone
import cv2
import math

# model = YOLOWorld('yolov8s-world.pt')
model = YOLO('yolov8n.pt')

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]
desired_features = ["person", "book", "cell phone", "laptop"]

def obj_detect(camera):
    ret, image = camera.read()
    if image is None:
        print("Error: Failed to capture frame")
        return {"person": None, "book": None, "cell phone": None, "laptop": None}
    
    # image = cv2.flip(image, 1)
    results = model.predict(image, device='cpu')

    count = {"person": 0, "book": 0, "cell phone": 0, "laptop": 0}

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Detect and process desired features
            if classNames[int(box.cls[0])] in desired_features:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(image, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(image, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1) 

                count[classNames[int(box.cls[0])]] += 1  # Increment count 
    cv2.imshow("Image", image)
    cv2.waitKey(1)
    return count
