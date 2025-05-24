import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder

import os
print(os.getcwd())

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'best_model.h5')
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

label_encoder = LabelEncoder()
label_encoder.fit(dataset['labels'])

labels_dict = {
    0: 'අ', 1: 'ආ', 2: 'ඇ', 3: 'ඉ',  4: 'ඊ',  5: 'උ',  6: 'ඌ',  7: 'එ',  8: 'ඒ',  9: 'ක්'
}
