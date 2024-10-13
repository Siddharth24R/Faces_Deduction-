import cv2
import os

# Create a directory to save the captured images
name = "siddharth"  # Replace with your name
data_dir =   r"C:\ALL folder in dexstop\PycharmProjects\face dedection\dataset\siddharth"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Initialize webcam and face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
    
    for (x, y, w, h) in faces:
        count += 1
        
        # Expand the region of interest (ROI) to capture a larger image
        padding = 40  # Adjust padding to increase the captured image size
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_img = gray[y1:y2, x1:x2]
        file_name = os.path.join(data_dir, f"{name}_{count}.jpg")
        cv2.imwrite(file_name, face_img)
        
        # Draw a larger rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (355, 0, 0), 2)
        cv2.putText(frame, f"Capturing {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Capture Images', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 500:  # Stop capturing after 500 images or pressing 'q'
        break

cap.release()
cv2.destroyAllWindows()
