"""
This script trains a 2D CNN on voxel-by-time fMRI matrices for face vs. no-face classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten, LeakyReLU)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import pickle
import matplotlib.pyplot as plt

#GPU Setup
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} Physical GPUs, {len(tf.config.list_logical_devices('GPU'))} Logical GPUs")
    except RuntimeError as e:
        print(e)


#Load Data
BASE_PATH = r"E:\Users\sara.asadi\pypro"
pickle_file = os.path.join(BASE_PATH, "fmri_data.pickle")

if not os.path.exists(pickle_file):
    raise FileNotFoundError(f"Pickle file not found: {pickle_file}")

print("Loading data from pickle...")
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
X_data, y_data = data['X'], data['y']

print(f"X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data, random_state=42)
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
print("Class weights:", class_weights)

#CNN Model
input_shape = X_train.shape[1:]

model = Sequential()

model.add(Conv2D(64, (5, 5), padding='same', kernel_regularizer=l2(0.002), input_shape=input_shape))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 1)))

model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.002)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 1)))
model.add(Dropout(0.3))

model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.002)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.002)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(128, kernel_regularizer=l2(0.002)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, kernel_regularizer=l2(0.002)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))


optimizer = Adam(learning_rate=0.0002)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#Training
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=18,
    class_weight=class_weights,
    verbose=1)


#Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

 
plt.figure(figsize=(12, 5))
 
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.xlim([0, epochs])
plt.ylim([0, 1])  #Adjust accuracy range

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.xlim([0, epochs])

plt.tight_layout()
plt.show()

#Save Model
model.save(os.path.join(BASE_PATH, 'model81.h5'))
model.save_weights(os.path.join(BASE_PATH, 'model81_weights.h5'))











