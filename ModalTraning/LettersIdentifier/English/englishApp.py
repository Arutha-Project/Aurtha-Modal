from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load model and label encoder
model = tf.keras.models.load_model('./best_model3.h5')

with open('./data.pickle', 'rb') as f:
    dataset = pickle.load(f)

labels = dataset[1].flatten()
label_encoder = LabelEncoder()
label_encoder.fit(labels)

labels_dict = {
    0: 'E', 1: 'B', 2: 'O', 3: 'L', 4: 'M', 5: 'N', 6: 'G', 7: 'P', 8: 'I', 9: 'R',
    10: 'A', 11: 'T', 12: 'C', 13: 'V', 14: 'K', 15: 'W', 16: 'X', 17: 'Y', 18: 'S', 19: 'D',
    20: 'E', 21: 'F', 22: 'G', 23: 'H', 24: 'Q', 25: 'Y'
}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.post("/predict-letter-english")
async def predict_letter(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            for lm in hand_landmarks.landmark:
                data_aux.append((lm.x - min(x_)) / (max(x_) - min(x_)))
                data_aux.append((lm.y - min(y_)) / (max(y_) - min(y_)))

            data_aux = np.array(data_aux).reshape(1, -1, 1)
            prediction = model.predict(data_aux)
            predicted_label_index = np.argmax(prediction)
            predicted_letter = labels_dict[predicted_label_index]

            return JSONResponse(content={"letter": predicted_letter})

    return JSONResponse(content={"letter": ""})
