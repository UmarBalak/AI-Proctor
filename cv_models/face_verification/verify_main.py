import cv2
import time
from verify_functions import *

camera = cv2.VideoCapture(0)
start_time = time.time()
flag = False
while time.time() - start_time < 180:
    # print(time.time() - start_time)  # Run for 180 seconds
    ret, frame = camera.read()
    count = 0
    for i in frame:
        count += len(i) 
    print(len(frame))
    print(count)
    verification = verify(frame)
    if verification:
        flag = True
        break
    else:
        continue
# Release the camera
camera.release()