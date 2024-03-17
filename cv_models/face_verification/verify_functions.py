# import cv2
# import face_recognition
# import numpy as np
# import time

# # Load a known face image
# known_image = face_recognition.load_image_file("tony.jpeg")
# known_face_encoding = face_recognition.face_encodings(known_image)[0]

# def verify(frame):
#     # Find all face encodings in the frame
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)

#     # Loop through each face found in the frame
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # Compare the face encoding with the known face encoding
#         matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

#         # If a match is found, set the label to "True"
#         if True in matches:
#             return True

#     return False



import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract features from an image using a pre-trained ResNet50 model
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Function to calculate cosine similarity between two feature vectors
def calculate_similarity(features1, features2):
    similarity = cosine_similarity([features1], [features2])[0][0]
    return similarity

# Load pre-trained ResNet50 model without the top classification layer
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Path to the two images
image1_path = 'img3.jpg'
image2_path = 'img4.png'

# Extract features for the two images
image1_features = extract_features(image1_path, model)
image2_features = extract_features(image2_path, model)

# Calculate similarity between the two images
similarity = calculate_similarity(image1_features, image2_features)
print(f"Similarity between {image1_path} and {image2_path}: {similarity}")
