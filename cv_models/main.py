from functions import *

while True:
    result = run()
    # if everything goes well, returns a python dictionary-->
    #    {"direction": direction, "distance": distance}
    # else returns a python dictionary-->
    #    {"face_detected": False}
    print(result)