import cv2
import mediapipe as mp
import time
import numpy as np
from numpy import greater
import utils
import math
import pandas as pd

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
    global frame_counter, CEF_COUNTER, TOTAL_BLINKS, frame_counter, eye_points

    # Left eyes indices 
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

    # right eyes indices
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

    frame_counter +=1 # frame counter
    if not ret: 
        return None, None, None, None  # no more frames break

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)

        l_eye_pts = []
        for i in range(0, 16):
            pt = mesh_coords[RIGHT_EYE[i]]
            l_eye_pts.append(pt)   
        # Draw a line connecting all points
        pts_array = np.array(l_eye_pts, np.int32)
        pts_array = pts_array.reshape((-1, 1, 2))

        r_eye_pts = []
        for i in range(0, 16):
            pt = mesh_coords[LEFT_EYE[i]]
            r_eye_pts.append(pt)
        # Draw a line connecting all points
        pts_array = np.array(r_eye_pts, np.int32)
        pts_array = pts_array.reshape((-1, 1, 2))
              
        ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

        if ratio > 5.5:
            CEF_COUNTER +=1
        else:
            if CEF_COUNTER > CLOSED_EYES_FRAME:
                TOTAL_BLINKS +=1
                CEF_COUNTER = 0

    else:
        return None, None, None, None

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
    else:
        return None, None, None, None

    direction = direction_estimator_1(r_eye_pts[8], r_eye_pts[0], center, 0.4, 0.3)
        
    # calculating  frame per seconds FPS
    end_time = time.time()-start_time
    fps = frame_counter/end_time
    
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
    else:
        return None
        

def calculate_distance(distance_pixel, distance_cm, success, image):
    # get correlation coefficients
    coff = np.polyfit(distance_pixel, distance_cm, 2)

    # perform face detection
    mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.75)
    while True:
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_face_detection.process(image)
        bbox_list, eyes_list = [], []
        if results.detections:
            for detection in results.detections:

                # get bbox data
                bboxc = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape
                bbox = int(bboxc.xmin*iw), int(bboxc.ymin *
                                                ih), int(bboxc.width*iw), int(bboxc.height*ih)
                bbox_list.append(bbox)

                # get the eyes landmark
                left_eye = detection.location_data.relative_keypoints[0]
                right_eye = detection.location_data.relative_keypoints[1]
                eyes_list.append([(int(left_eye.x*iw), int(left_eye.y*ih)),
                                  (int(right_eye.x*iw), int(right_eye.y*ih))])

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for bbox, eye in zip(bbox_list, eyes_list):

            # calculate distance between left and right eye
            dist_between_eyes = np.sqrt(
                (eye[0][1]-eye[1][1])**2 + (eye[0][0]-eye[1][0])**2)

            # calculate distance in cm
            a, b, c = coff
            distance_cm = a*dist_between_eyes**2+b*dist_between_eyes+c
            distance_cm -= 7
            
            return distance_cm
        
        else:
            return None



def run():
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)

    blink_ratio, total_blinks, eye_direction, fps = eye_track(ret, frame, rgb_frame, results)
    if blink_ratio is None or total_blinks is None or eye_direction is None:
        # case where no face is detected
        blink_ratio = 0
        total_blinks = 0
        eye_direction = "No face detected"
        fps = 0

    head_direction = head_pose(rgb_frame, results)
    if head_direction is None:
        # case where no face is detected
        head_direction = "No face detected"

    distance_df = pd.read_csv('distance_xy.csv')
    distance_pixel = distance_df['distance_pixel'].tolist()
    distance_cm = distance_df['distance_cm'].tolist()
    distance_cm = calculate_distance(distance_pixel, distance_cm, ret, frame)

    end = time.time()
    totalTime = end - start_time  
    if totalTime > 0:
        fps = 1 / totalTime
    else:
        fps = 0

    return {"blink_ratio": blink_ratio, "total_blinks": total_blinks, "eye_direction": eye_direction, "head_direction": head_direction, "distance": distance_cm, "fps": fps}
