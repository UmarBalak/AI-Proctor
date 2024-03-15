from yolo_functions import *

camera = cv2.VideoCapture(0)

while True:
    counts = obj_detect(camera)
    print(counts)


# camera.release()
# cv2.destroyAllWindows()