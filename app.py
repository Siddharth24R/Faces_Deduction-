from flask import Flask, render_template, Response
import cv2
import numpy as np
import face_recognition
import threading
import datetime
import requests
from gtts import gTTS
import os
import pygame
import time

# Initialize pygame mixer for gTTS
pygame.mixer.init()

# Initialize Flask app
app = Flask(__name__)

# Pre-generate the greeting message as MP3 to reduce delay
def generate_greeting_message():
    greeting = get_greeting()
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    weather = get_weather()
    message = f"Hi Siddharth, {greeting}. The current time is {current_time}. {weather}. Call me Riya for any assistance."
    tts = gTTS(text=message, lang='en')
    tts.save("greeting.mp3")

# Function to speak pre-generated greeting
def speak_greeting():
    pygame.mixer.music.load("greeting.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Function to get the weather (using OpenWeatherMap API)
def get_weather():
    api_key = "a82a2b8b84989b42d17d4807d8eef400" # Replace with your OpenWeatherMap API key
    city = "chennai"  # Replace with your city
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    if response["cod"] == 200:
        weather = response["weather"][0]["description"]
        temperature = response["main"]["temp"]
        return f"The current weather in {city} is {weather} with a temperature of {temperature}°C."
    else:
        return "Sorry, I couldn't retrieve the weather."

# Get appropriate greeting based on the time
def get_greeting():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good Morning"
    elif hour < 18:
        return "Good Afternoon"
    else:
        return "Good Evening"

# Pre-generate the greeting message on program start
generate_greeting_message()

# Load the encodings, ids, and names
encodings_file = r"C:\\ALL folder in dexstop\\PycharmProjects\\face dedection\\npy_files\\encodings.npy"
ids_file = r"C:\\ALL folder in dexstop\\PycharmProjects\\face dedection\\npy_files\\ids.npy"
names_file = r"C:\\ALL folder in dexstop\\PycharmProjects\\face dedection\\npy_files\\names.npy"

try:
    data = np.load(encodings_file)
    ids = np.load(ids_file)
    names = np.load(names_file)
    print("Encodings loaded successfully.")
except Exception as e:
    print(f"Error loading encodings: {e}")
    exit()

# Load the Haar Cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Adjust threshold for improved accuracy
threshold = 0.4  # Lower threshold for stricter identification
buffer_zone = 0.1  # Buffer zone between confirmed Siddharth and unknown
siddharth_greeted = False  # To check if Siddharth has been greeted

def gen_frames():
    global siddharth_greeted
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to RGB (required by face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection using Haar Cascade
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        print(f"Detected faces: {len(faces)}")

        if len(faces) == 0:
            print("No face detected.")
            siddharth_greeted = False  # Reset greeted flag if no face is detected
        
        for (x, y, w, h) in faces:
            # Define the region of interest (ROI) for face
            face_roi = rgb_frame[y:y + h, x:x + w]
            face_encodings = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])
            
            if face_encodings:
                face_encoding = face_encodings[0]
                distances = np.linalg.norm(data - face_encoding, axis=1)
                min_distance = np.min(distances)

                print(f"Detected min_distance: {min_distance}")

                if min_distance < threshold:
                    name = "Siddharth"
                    color = (0, 255, 0)  # Green color for recognized face
                    
                    if not siddharth_greeted:
                        threading.Thread(target=speak_greeting).start()
                        siddharth_greeted = True

                elif threshold <= min_distance < (threshold + buffer_zone):
                    name = "Unsure"
                    color = (255, 255, 0)  # Yellow for unsure face
                    print("Unsure if it's Siddharth or not.")

                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown face
                    siddharth_greeted = False

                # Draw rectangle around detected face and display name
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Encode the frame in JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
