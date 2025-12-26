import numpy as np
import cv2
from ModalTraning.LettersIdentifier.Sinhala.model_loader import model, labels_dict
from Util.mediapipe_utils import hands
from fastapi import UploadFile
import base64

async def predict_letter_sinhala(file: UploadFile) -> str:
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return ""

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        letters = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness_label = results.multi_handedness[idx].classification[0].label  # "Left" or "Right"

            if handedness_label == "Right":
                x_ = [1 - lm.x for lm in hand_landmarks.landmark]
            else:
                x_ = [lm.x for lm in hand_landmarks.landmark]

            y_ = [lm.y for lm in hand_landmarks.landmark]

            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                normalized_x = (x_[i] - min(x_)) / (max(x_) - min(x_) + 1e-6)
                normalized_y = (y_[i] - min(y_)) / (max(y_) - min(y_) + 1e-6)
                data_aux.append(normalized_x)
                data_aux.append(normalized_y)

            data_aux = np.array(data_aux).reshape(1, -1, 1)
            prediction = model.predict(data_aux)
            predicted_label_index = np.argmax(prediction)
            predicted_letter = labels_dict.get(predicted_label_index, "")
            letters.append(predicted_letter)

        return " ".join(letters)

    return ""


async def predict_letter_sinhala_activity(frame_base64: str) -> str:
    try:
        frame_data = base64.b64decode(frame_base64)
        np_arr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            print("⚠️ Failed to decode image")
            return ""

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            letters = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_label = results.multi_handedness[idx].classification[0].label  # "Left" or "Right"

                if handedness_label == "Right":
                    x_ = [1 - lm.x for lm in hand_landmarks.landmark]
                else:
                    x_ = [lm.x for lm in hand_landmarks.landmark]

                y_ = [lm.y for lm in hand_landmarks.landmark]

                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    normalized_x = (x_[i] - min(x_)) / (max(x_) - min(x_) + 1e-6)
                    normalized_y = (y_[i] - min(y_)) / (max(y_) - min(y_) + 1e-6)
                    data_aux.append(normalized_x)
                    data_aux.append(normalized_y)

                data_aux = np.array(data_aux).reshape(1, -1, 1)
                prediction = model.predict(data_aux)
                predicted_index = np.argmax(prediction)
                predicted_letter = labels_dict.get(predicted_index, "Unknown")
                letters.append(predicted_letter)

            return " ".join(letters)

        return {"error": "No hand detected"}
    
    except Exception as e:
        print("Error:", e)
        return {"error": "Server error"}
