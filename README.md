
# Flask Face Recognition and Voice Assistant


This Flask application uses face recognition to identify Siddharth and provides a personalized greeting, including the current time and weather information. The application integrates OpenCV for face detection, face_recognition for identification, and gTTS for voice greetings.


## Features

- **Real-time Face Detection**: Uses OpenCV's Haar Cascade to detect faces in the webcam feed.
- **Face Recognition**: Identifies the user by comparing live video feed face encodings to pre-trained face encodings.
- **Voice Greeting**: Plays a personalized greeting message, including the current time and weather, when the user is recognized.
- **Weather Information**: Fetches current weather conditions using the OpenWeatherMap API.
- **Web Interface**: The application is accessible via a web browser, displaying the live video feed.

## Requirements

Make sure you have the following installed:

- Python 3.x
- Flask
- OpenCV (`cv2`)
- face_recognition
- numpy
- gTTS (Google Text-to-Speech)
- pygame
- requests

### Python Libraries Installation

You can install the required Python libraries with the following command:

```bash
pip install Flask opencv-python face_recognition numpy gTTS pygame requests
```

## Project Structure

```
.
├── app.py               # Main Flask application
├── templates
│   └── index.html       # HTML template for video feed
├── greeting.mp3         # Pre-generated greeting message (generated at runtime)
├── npy_files
│   ├── encodings.npy    # Encoded face data
│   ├── ids.npy          # IDs corresponding to the face encodings
│   └── names.npy        # Names of users corresponding to IDs
└── README.md            # Project documentation
```

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Prepare Face Encodings**:

   Before running the app, make sure to generate face encodings for the users you want to recognize. You can create these `.npy` files using the `face_recognition` library.

3. **Set OpenWeatherMap API Key**:

   Replace the `api_key` variable in `app.py` with your own OpenWeatherMap API key.

4. **Run the Flask App**:

   You can start the application with the following command:

   ```bash
   python app.py
   ```

   The application will be available at `http://0.0.0.0:5000/` by default.

5. **Access the App**:

   Open a browser and go to `http://localhost:5000/` to see the live video feed from your webcam. The app will recognize a known user and greet them with a personalized voice message.

## Customization

- **Add More Users**: To add more users for face recognition, update the `encodings.npy`, `ids.npy`, and `names.npy` files with additional face data.
- **Weather Location**: Change the city in the `get_weather()` function in `app.py` to retrieve weather data for a different location.

## Troubleshooting

- Ensure your webcam is working properly.
- Make sure the `encodings.npy`, `ids.npy`, and `names.npy` files are in the correct directory and contain valid data.
- Verify that your OpenWeatherMap API key is correct and the city name is valid.

## Acknowledgments

- [OpenCV](https://opencv.org/) for face detection.
- [face_recognition](https://github.com/ageitgey/face_recognition) library for face recognition.
- [gTTS](https://pypi.org/project/gTTS/) for Google Text-to-Speech integration.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
