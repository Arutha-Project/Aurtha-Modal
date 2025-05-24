from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

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
model = tf.keras.models.load_model('./best_model3.h5')

with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data, labels = data_dict 
labels = labels.flatten() 

label_encoder = LabelEncoder()
label_encoder.fit(labels)

labels_dict = {
    0: 'E', 1: 'B', 2: 'O', 3: 'L', 4: 'M', 5: 'N', 6: 'G', 7: 'P', 8: 'I', 9: 'R',
    10: 'A', 11: 'T', 12: 'C', 13: 'V', 14: 'K', 15: 'W', 16: 'X', 17: 'Y', 18: 'S', 19: 'D',
    20: 'E', 21: 'F', 22: 'G', 23: 'H', 24: 'Q', 25: 'Y'
}

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.post("/predict-activity/english-letter")
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

