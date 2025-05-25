import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder

import os
print(os.getcwd())

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'best_model3.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
# Load model
model = tf.keras.models.load_model(model_path)

pickle_path = os.path.join(base_path, 'data.pickle')
if not os.path.exists(pickle_path):
    raise FileNotFoundError(f"Pickle file not found at {pickle_path}")
# Load labels
with open(pickle_path, 'rb') as f:
    dataset = pickle.load(f)

labels = dataset[1].flatten()
label_encoder = LabelEncoder()
label_encoder.fit(labels)

labels_dict = {
    0: 'E', 1: 'B', 2: 'O', 3: 'L', 4: 'M', 5: 'N', 6: 'G', 7: 'P', 8: 'I', 9: 'R',
    10: 'A', 11: 'T', 12: 'C', 13: 'V', 14: 'K', 15: 'W', 16: 'X', 17: 'Y', 18: 'S', 19: 'D',
    20: 'E', 21: 'F', 22: 'G', 23: 'H', 24: 'Q', 25: 'Y'
}
