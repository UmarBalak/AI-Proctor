from yolo_functions import *

camera = cv2.VideoCapture(0)

while True:
    count = obj_detect(camera)
    print(count)