import cv2
import mediapipe as mp
import time
import numpy as np
from numpy import greater
import utils
import math

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

# variables 
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
frame_counter =0

# constants 
CLOSED_EYES_FRAME =1
FONTS =cv2.FONT_HERSHEY_COMPLEX

map_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255),thickness=1,circle_radius=1)

camera = cv2.VideoCapture(0)

start_time = time.time()


def speak(text):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]
    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    
    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio

def direction_estimator_1(extreme_right_circle_right_eye, extreme_left_circle_right_eye, gaze_center, l_eye_threshold, r_eye_threshold):
    # input :- takes 3 tuples
    # output :- returns the direction
    dist_gaze_and_rightOfRight = extreme_right_circle_right_eye[0] - gaze_center[0]
    dist_gaze_and_leftOfRight = gaze_center[0] - extreme_left_circle_right_eye[0]
    eye_width = extreme_right_circle_right_eye[0] - extreme_left_circle_right_eye[0]
    if dist_gaze_and_rightOfRight < (eye_width * r_eye_threshold):
        direction = "right"
    elif dist_gaze_and_leftOfRight < (eye_width * l_eye_threshold):
        direction = "left"
    else:
        direction = "center"
    return direction

def direction_estimator_2(r_eye_pts, gaze_center):
    distance = {}
    for i in range(0, 16):
        dist = abs(gaze_center[0] - r_eye_pts[i][0])
        distance[i] = dist
    top_5_smallest = sorted(distance.items(), key=lambda x: x[1])[:5]
    print(top_5_smallest)
    keys = [item[0] for item in top_5_smallest]
    required_keys_for_left = {2, 3, 13, 14}
    required_keys_for_right = {6, 7, 10, 11}
    if required_keys_for_left.issubset(keys):
        direction = "left"
    elif required_keys_for_right.issubset(keys):
        direction = "right"
    else:
        direction = "center"
    return direction


def points_on_circle(center, radius, num_points):
    points = []
    for i in range(num_points):
        angle = i * (2 * np.pi / num_points)
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append((x, y))
    return points


def draw_sharingan(frame, center, radius):
    sharingan_clr = (19, 19, 175)
    black_clr = (9, 9, 9)
    cv2.circle(frame, center, radius, black_clr, int(radius * 0.125))

    # Extra cirlces just for fun
    cv2.circle(frame, center, int(radius * 0.875), sharingan_clr, int(radius * 0.15)) 
    cv2.circle(frame, center, int(radius * 0.725), sharingan_clr, int(radius * 0.15)) # sharingan points
    cv2.circle(frame, center, int(radius * 0.575), sharingan_clr, int(radius * 0.15))
    cv2.circle(frame, center, int(radius * 0.425), black_clr, int(radius * 0.1)) 
    cv2.circle(frame, center, int(radius * 0.325), sharingan_clr, -1)   

    # Get points on the border of the circle (sharingan points)
    border_points = points_on_circle(center, int(radius * 0.5), 3)
    # Draw the sharingan points
    for point in border_points:
        cv2.circle(frame, point, int(radius * 0.075) + 1, black_clr, -1)


def draw_mesh(frame, r_eye_pts, gaze_center):
    distance = {}
    for i in range(0, 16):
        dist = abs(gaze_center[0] - r_eye_pts[i][0])
        cv2.line(frame, gaze_center, r_eye_pts[i], (19, 19, 175), 1)
        distance[i] = dist


def eye_track(ret, frame, rgb_frame, results):
    global frame_counter, CEF_COUNTER, TOTAL_BLINKS, frame_counter
    frame_counter +=1 # frame counter
    if not ret: 
        eye_track()  # no more frames break

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)

        l_eye_pts = []
        for i in range(0, 16):
            pt = mesh_coords[RIGHT_EYE[i]]
            l_eye_pts.append(pt)   
#         for i in range(0, 16):
#             cv2.circle(frame, l_eye_pts[i], 1, (19, 19, 175), -1, cv2.LINE_AA)
        # Draw a line connecting all points
        pts_array = np.array(l_eye_pts, np.int32)
        pts_array = pts_array.reshape((-1, 1, 2))
#         cv2.polylines(frame, [pts_array], isClosed=False, color=(19, 19, 175), thickness=1, lineType=cv2.LINE_AA)

        r_eye_pts = []
        for i in range(0, 16):
            pt = mesh_coords[LEFT_EYE[i]]
            r_eye_pts.append(pt)
#         for i in range(0, 16):
#             cv2.circle(frame, r_eye_pts[i], 1, (19, 19, 175), -1, cv2.LINE_AA)
        # Draw a line connecting all points
        pts_array = np.array(r_eye_pts, np.int32)
        pts_array = pts_array.reshape((-1, 1, 2))
#         cv2.polylines(frame, [pts_array], isClosed=False, color=(19, 19, 175), thickness=1, lineType=cv2.LINE_AA)
              
        ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
#         utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

        if ratio > 5.5:
            CEF_COUNTER +=1
#             utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
        else:
            if CEF_COUNTER>CLOSED_EYES_FRAME:
                TOTAL_BLINKS +=1
                CEF_COUNTER =0

#         utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
########################################################################################################################   
    
    frame_h, frame_w, _ = frame.shape
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    if landmark_points:
        landmarks = landmark_points[0].landmark   
        
        # Get coordinates of the four points around the eye
        eye_points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in range(474, 478)]   
    
    # Draw circle approximating the eye
    if len(eye_points) == 4:
        center, radius = cv2.minEnclosingCircle(np.array(eye_points))
        center = (int(center[0]), int(center[1]))
        radius = int(radius * 0.75)
        # draw_sharingan(frame, center, radius)            
            
    direction = direction_estimator_1(r_eye_pts[8], r_eye_pts[0], center, 0.4, 0.3)
#     direction = direction_estimator_2(extreme_right_circle_right_eye, extreme_left_circle_right_eye, center)
        
    # calculating  frame per seconds FPS
    end_time = time.time()-start_time
    fps = frame_counter/end_time
    # cv2.imshow('frame', frame)
    # key = cv2.waitKey(1)
    
    return ratio, TOTAL_BLINKS, direction, fps


def head_pose(frame, results):
    
    img_h,img_w,Img_c = frame.shape
    face_3d = []
    face_2d = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w , lm.y * img_h)
                        nose_3d = (lm.x * img_w , lm.y * img_h , lm.z * 3000)
                    
                    x , y = int(lm.x * img_w) , int(lm.y * img_h)
                    
                    face_2d.append([x, y])
                    
                    face_3d.append([x, y, lm.z])
                    
            face_2d = np.array(face_2d, dtype = np.float64)
            
            face_3d = np.array(face_3d, dtype = np.float64)
            
            focal_length = 1 * img_w
            
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                  [0 ,focal_length, img_w / 2],
                                  [0 ,0 ,1]])
            
            dist_matrix = np.zeros((4,1),dtype= np.float64)
            
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            rmat, jac = cv2.Rodrigues(rot_vec)
            
            angles , mtxR , mtxQ , Qx , Qy , Qz = cv2.RQDecomp3x3(rmat)
            
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            if y < -18:
                text = "Looking Left"
            elif y > 20:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 20:
                text = "Looking Up"
            else:
                text = "Forward"
            
            return text
        

def run():
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)
    blink_ratio, total_blinks, eye_direction, fps = eye_track(ret, frame, rgb_frame, results)
    head_direction = head_pose(rgb_frame, results)
    end = time.time()
    totalTime = end - start_time
    
    if totalTime > 0:
        fps = 1 / totalTime
    else:
        fps = 0
    return {"blink_ratio": blink_ratio, "total_blinks": total_blinks, "eye_direction": eye_direction, "head_direction": head_direction, "fps": fps}
