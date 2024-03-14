from functions import *
while True:
    output = run()
    # if everything goes well, returns a python dictionary-->
    #    {"direction": direction, "distance": distance}
    # else returns a python dictionary-->
    #    {"face_detected": False}
    if output["termination"] is True:
        print(output["result"])
        break
    print(output["result"])