from functions import *

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    count = obj_detect(frame)
    print(count)