from functions import *

camera = cv2.VideoCapture(0)
count = 0
while True:
    result = run(camera)
    print(result)
    if not result:
        count += 1
    if  count == 4:
        break
    elif count == 3:
        speak("This is the last warning, After this, your exam will be terminated")


# camera.release()
# cv2.destroyAllWindows()