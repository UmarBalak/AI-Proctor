import cv2
import face_recognition
import numpy as np

# Load a known face image
known_image = face_recognition.load_image_file("tony.jpeg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

def verify(frame):
    # Find all face encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with the known face encoding
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

        # If a match is found, set the label to "True"
        if True in matches:
            return True

    return False