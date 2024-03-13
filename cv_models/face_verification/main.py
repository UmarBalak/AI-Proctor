from functions import *
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    verification = verify(frame)
    if verification is True:
        print("Candidate Verified.")
        break
    else:
        continue

