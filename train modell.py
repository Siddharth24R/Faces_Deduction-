import face_recognition
import cv2
import os
import numpy as np

# Directory containing training images
train_dir = 'C:\\ALL folder in dexstop\\PycharmProjects\\face dedection\\dataset\\siddharth'

# Initialize the arrays to store known face encodings and corresponding labels
known_face_encodings = []
known_face_names = []

# Loop through each person in the training directory
for person_name in os.listdir(train_dir):
    person_dir = os.path.join(train_dir, Siddharth)
    
    # Loop through each image of the person
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        
        # Load the image and get the face encoding
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        # Only proceed if at least one face is detected in the image
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(person_name)

print("Training complete!")
