from functions import *

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    print(frame)
    count = obj_detect(frame)
    print(count)