from functions import *

camera = cv2.VideoCapture(0)

while True:
    result = obj_detect(camera)
    if result is True:
        print("okk")
    else:
        print("Alert")
        break


# camera.release()
# cv2.destroyAllWindows()