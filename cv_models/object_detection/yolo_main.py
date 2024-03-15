from yolo_functions import *

camera = cv2.VideoCapture(0)

while True:
    result = obj_detect(camera)
    print(result)   