from functions import *
count = 0
while True:
    result = run()
    # if everything goes well, returns a python dictionary-->
    #    {"direction": direction, "distance": distance}
    # else returns a python dictionary-->
    #    {"face_detected": False}
    
    if len(result) == 2:
        if result['direction'] == 'Right' or  result['direction'] == 'Left':
            count += 1
        print(count)
    else:
        print(result)