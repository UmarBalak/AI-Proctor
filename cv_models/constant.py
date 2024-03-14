import cv2
import mediapipe as mp
import time
import numpy as np
from numpy import greater
import utils
import math
import pandas as pd
import pyttsx3

# variables for direction alert
change_dir_counter = 0
dir_warning_counter = 0
vis_warning_counter = 0
warning_count = 0
visibility_counter = 0

# variables 
frame_counter =0
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