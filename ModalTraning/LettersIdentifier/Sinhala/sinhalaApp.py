fixedfrom fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('./best_model.h5')

# Load label encoder
with open('./data.pickle', 'rb') as f:
    dataset = pickle.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(dataset['labels'])

# Dictionary for Sinhala letter mapping
labels_dict = {
    0: 'අ', 1: 'ආ', 2: 'ඇ', 3: 'ඉ',  4: 'ඊ',  5: 'උ',  6: 'ඌ',  7: 'එ',  8: 'ඒ',  9: 'ක්'
}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.post("/predict-letter-sinhala")
async def predict_sinhala_letter(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # If hands detected, make prediction
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append((lm.x - min(x_)) / (max(x_) - min(x_)))
                data_aux.append((lm.y - min(y_)) / (max(y_) - min(y_)))

            data_aux = np.array(data_aux).reshape(1, -1, 1)
            prediction = model.predict(data_aux)
            predicted_label_index = np.argmax(prediction)
            predicted_letter = labels_dict.get(predicted_label_index, "")

            return JSONResponse(content={"letter": predicted_letter})

    return JSONResponse(content={"letter": ""})
