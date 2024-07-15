import os.path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Charger le modèle pré-entraîné MobileNetV2
model = MobileNetV2(weights='imagenet')

# Fonction pour charger et préparer une image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Prédire la classe de l'image
def predict_image_class(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Sauvegarde du modèle en SavedModel (nécessaire pour la conversion TFLite)
tf.saved_model.save(model, "model")

# Conversion vers TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model") # path to the SavedModel directory
tflite_model = converter.convert()

# Sauvegarde du modèle TFLite
with open('model_lite/model.tflite', 'wb') as f:
  f.write(tflite_model)

# Exemple d'utilisation
img_path = 'img/pq.jpg'
print(predict_image_class(img_path))
