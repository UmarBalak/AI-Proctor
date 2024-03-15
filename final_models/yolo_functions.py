from ultralytics import YOLO
from ultralytics import YOLOWorld
import cvzone
import cv2
import math

# Load the YOLO model
# model = YOLOWorld('yolov8s-world.pt')
# model = YOLO('yolov8m.pt')
model = YOLO('yolov8s.pt')

# Define the class names
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
# Define the desired features
desired_features = ["person", "book", "cell phone", "laptop"]

from datetime import datetime, timedelta

# Define the maximum duration before triggering the alert (in seconds)
MAX_ALERT_DURATION = 5  # Adjust as needed

# Initialize the timer
alert_timer = 0
alert_triggered = False

def obj_detect(camera):
    global alert_timer, alert_triggered
    
    ret, image = camera.read()
   
    # Flip the image horizontally
    image = cv2.flip(image, 1)
    
    # Perform object detection
    results = model.predict(image, device='cpu')

    # Initialize count list for desired features
    count = [0] * len(desired_features)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Get the class ID
            cls_id = int(box.cls[0])
            
            # Get the class name
            class_name = classNames[cls_id]
            
            # Check if the detected class is one of the desired features
            if class_name in desired_features:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
                w, h = x2 - x1, y2 - y1  # Calculate width and height of the bounding box
                
                # Draw the bounding box and label
                cvzone.cornerRect(image, (x1, y1, w, h))
                conf = round(box.conf[0].item(), 2) # Round confidence score to two decimal places
                cvzone.putTextRect(image, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1) 
                
                # Increment count for the detected class
                count[desired_features.index(class_name)] += 1  
    
    # Check if person count exceeds threshold
    if count[desired_features.index("person")] > 2:
        # Increment the alert timer
        alert_timer += 1
        if alert_timer > MAX_ALERT_DURATION:
            # Trigger alert
            alert_triggered = True
            alert_timer = 0  # Reset the timer after alert

    # Display the image
    # cv2.imshow("Image", image)
    # cv2.waitKey(1)
    
    return count, alert_triggered



