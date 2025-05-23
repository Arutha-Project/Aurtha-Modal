from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import json

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and label info
model = load_model('./best_model.h5')

with open('./data.pickle', 'rb') as f:
    dataset = pickle.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(dataset['labels'])

# Dictionary for Sinhala letter mapping
labels_dict = {0: 'අ', 1: 'ආ', 2: 'ඇ', 3: 'ඉ',  4: 'ඊ',  5: 'උ',  6: 'ඌ',  7: 'එ',  8: 'ඒ',  9: 'ක්'}

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.post("/predict-activity/sinhala-letter")
async def predict_letter(payload: dict):
    try:
        frame_data = base64.b64decode(payload["frame"])
        np_arr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            print("Image decode failed.")
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            print("Hand detected.")
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_, data_aux = [], [], []
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append((lm.x - min(x_)) / (max(x_) - min(x_)))
                    data_aux.append((lm.y - min(y_)) / (max(y_) - min(y_)))

                data_aux = np.array(data_aux).reshape(1, -1, 1)
                prediction = model.predict(data_aux)
                predicted_label_index = np.argmax(prediction)
                predicted_letter = labels_dict.get(predicted_label_index, "Unknown")


                # print(f"Predicted: {predicted_letter}")
                print(f"🎯 Predicted index: {predicted_label_index}, Letter: {predicted_letter}")

                return JSONResponse(content={"predicted_letter": predicted_letter})

        print("No hand landmarks found.")

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
          print("⚠️ Failed to decode image")
          return JSONResponse(content={"error": "Invalid image"}, status_code=400)
        else:
          print("✅ Image decoded successfully")

        return JSONResponse(content={"error": "No hand detected"}, status_code=400)

    except Exception as e:
        print("Error in prediction:", e)
        return JSONResponse(content={"error": "Server error"}, status_code=500)

