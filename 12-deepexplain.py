# This code Builds a logits view directly from the last layer
import os
import warnings
import numpy as np
import gc
import tensorflow.compat.v1 as tf1
import tensorflow as tf

#   TensorFlow 1.x compatibility patch  
if not hasattr(tf, "get_default_session"):
    tf.get_default_session = tf1.get_default_session
if not hasattr(tf, "get_default_graph"):
    tf.get_default_graph = tf1.get_default_graph
if not hasattr(tf, "placeholder"):
    tf.placeholder = tf1.placeholder

from deepexplain.tensorflow import DeepExplain

# Silence TensorFlow v2 warnings & disable eager
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
tf1.disable_v2_behavior()
tf1.disable_eager_execution()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, LeakyReLU, Flatten
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

K.set_learning_phase(0)  # inference mode (disable dropout/bn training)

# CONFIG  
label = "face"

root_dir = rf"D:\sarafiles\Face-Project\data-npy-seperately-for-deepexplain-one-by-one\{label}"
save_root = rf"D:\sarafiles\Face-Project\IG\{label}"
os.makedirs(save_root, exist_ok=True)


model_weights = r"D:\sarafiles\Face-Project\best-result-cnn\model83_weights-new.h5"

# MODEL DEFINITION  
def get_model():
    input_shape = (41489, 10, 1)
    inputs = Input(shape=input_shape, name="inputs")

    x = Conv2D(64, (5, 3), padding='same', kernel_initializer="he_normal",
               kernel_regularizer=l2(0.001))(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(32, (5, 3), padding='same', kernel_initializer="he_normal",
               kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.35)(x)

    x = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal",
               kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128, kernel_initializer="he_normal", kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    outputs = Dense(1, activation='sigmoid', name="outputs")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0002),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

#   LOAD MODEL  
model = get_model()
model.load_weights(model_weights)

last = model.get_layer('outputs')          # Dense(1, activation='sigmoid')
prev = last.input                          # features into Dense
logits = tf.matmul(prev, last.kernel) + last.bias  # W·x + b  → the logit


for sub_folder in sorted(os.listdir(root_dir)):
    sub_path = os.path.join(root_dir, sub_folder)
    if not os.path.isdir(sub_path):
        continue

    print(f"\nProcessing {sub_folder}...")

    save_dir = os.path.join(save_root, sub_folder)
    os.makedirs(save_dir, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(sub_path) if f.endswith(".npy")])

    for file in npy_files:
        subj_name = os.path.splitext(file)[0]
        file_path = os.path.join(sub_path, file)
        print(f"  Loading {file_path} ...")

        x_data = np.load(file_path).astype(np.float32)
        if x_data.ndim == 2:
            x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=0)

        baseline_single = np.zeros((41489, 10, 1), dtype=np.float32)
    
        if label.lower() == "face":
            target_T = logits                # Face = class 1 → logit
        else:
            target_T = tf.math.negative(logits)  # NoFace = class 0 → -logit

        with DeepExplain(session=tf1.keras.backend.get_session()) as de:
            attributions = de.explain(
                method='intgrad',
                T=target_T,
                X=model.input,
                xs=x_data,
                baseline=baseline_single,
                steps=200
            )

        save_path = os.path.join(save_dir, f"IG_{subj_name}.npy")
        np.save(save_path, attributions.astype(np.float32))
        print(f"  → Saved IG to: {save_path}")

    print(f"Completed {sub_folder}.")

gc.collect()
print("\n All subjects processed.")
