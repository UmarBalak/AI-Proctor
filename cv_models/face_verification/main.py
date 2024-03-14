import cv2
import time
from functions import *

camera = cv2.VideoCapture(0)
start_time = time.time()

while time.time() - start_time < 180:
    # print(time.time() - start_time)  # Run for 180 seconds
    ret, frame = camera.read()
    verification = verify(frame)
    if verification:
        print("Candidate Verified.")
        break
    else:
        continue

# Release the camera
camera.release()

# If 30 seconds have elapsed and verification is not successful, print "False"
if not verification:
    print("Unable to verify.")
