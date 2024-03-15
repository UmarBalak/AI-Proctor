from functions import *
camera = cv2.VideoCapture(0)
while True:
    output = run()
    # if everything goes well, returns a python dictionary-->
    #    {"direction": direction, "distance": distance}
    # else returns a python dictionary-->
    #    {"face_detected": False}

    print(output)

 