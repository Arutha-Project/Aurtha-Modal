from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS  # Import the CORS package
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ImageFont
import io

# Initialize the Flask app
app = Flask(__name__)
socketio = SocketIO(app=app, cors_allowed_origins="*")

# Enable CORS for the app
#CORS(app, origins="*")  # Allow requests from React frontend (localhost:5174)

# Load the trained model
model = load_model('./best_model3.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Label encoder for decoding predictions
with open('./data.pickle', 'rb') as f:
    dataset = pickle.load(f)

labels = dataset[1]  # Assuming labels are the second element in the tuple
labels = labels.flatten()  # Flatten the labels array to 1D
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Dictionary for Sinhala letter mapping
labels_dict = {
    0: 'E', 1: 'B', 2: 'O', 3: 'L', 4: 'M', 5: 'N', 6: 'G', 7: 'P', 8: 'I', 9: 'R',
    10: 'A', 11: 'T', 12: 'C', 13: 'V', 14: 'K', 15: 'W', 16: 'X', 17: 'Y', 18: 'S', 19: 'D',
    20: 'E', 21: 'F', 22: 'G', 23: 'H', 24: 'Q', 25: 'Y'
}

# Socket.IO event to receive frames from the frontend
@socketio.on('frame')
def handle_frame(frame_data):
    # Convert the incoming frame (from base64) into an OpenCV image
    img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Process the image using MediaPipe
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Normalize landmarks for prediction
            data_aux = []
            x_ = []
            y_ = []
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append((lm.x - min(x_)) / (max(x_) - min(x_)))
                data_aux.append((lm.y - min(y_)) / (max(y_) - min(y_)))

            # Make prediction
            data_aux = np.array(data_aux).reshape(1, -1, 1)
            prediction = model.predict(data_aux)
            predicted_label_index = np.argmax(prediction)
            predicted_letter = labels_dict[predicted_label_index]

            # Emit the predicted letter to the frontend
            emit('predicted_letter', predicted_letter)

            # Convert frame back to PIL and add text
            draw = ImageDraw.Draw(pil_img)
            draw.text((50, 50), f'Letter: {predicted_letter}', font=ImageFont.load_default(), fill=(255, 0, 0))

    # Convert the frame back to OpenCV format
    frame = np.array(pil_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Convert the processed frame back to base64 to send to frontend
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = buffer.tobytes()

    # Emit the processed frame back to the frontend
    emit('processed_frame', frame_data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)