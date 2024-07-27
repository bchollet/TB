import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Chemins vers les données
train_dir = './dataset/merged/train'
valid_dir = './dataset/merged/valid'

# Lecture des fichiers _classes.csv
train_classes = pd.read_csv(os.path.join(train_dir, '_classes.csv'), header=None, names=['filename', 'class'])
valid_classes = pd.read_csv(os.path.join(valid_dir, '_classes.csv'), header=None, names=['filename', 'class'])

# Préparation des générateurs de données
train_datagen = ImageDataGenerator(rescale=0.017, validation_split=0.2)
valid_datagen = ImageDataGenerator(rescale=0.017)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_classes,
    directory=train_dir,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_classes,
    directory=valid_dir,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Chargement du modèle MobileNetV3 avec les poids ImageNet
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle avec suivi
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    epochs=10
)

# Sauvegarde du modèle en SavedModel
tf.saved_model.save(model, "model")

# Plot de la progression de l'entraînement
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()