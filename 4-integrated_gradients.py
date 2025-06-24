"""
Integrated Gradients Computation for fMRI Face Classification
==============================================================
This script loads preprocessed fMRI data, restores a trained CNN model, and computes Integrated Gradients (IG) using DeepExplain for interpretability.
It targets the pre-sigmoid logit to avoid gradient saturation and saves per-subject IG maps.

"""


import os
import pickle
import warnings
import numpy as np
import tensorflow.compat.v1 as tf1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from deepexplain.tensorflow import DeepExplain

# Silence TensorFlow v2 warnings
warnings.filterwarnings("ignore")
tf1.disable_v2_behavior()
tf1.disable_eager_execution()

# Configurable labels and directories
label = "face"
pickle_path = rf"E:\sara.asadi\data-pickled-seperately-for-deepexplain\{label}"
save_dir    = rf"E:\sara.asadi\IG-pre‐sigmoid-logit\{label}"
os.makedirs(save_dir, exist_ok=True)

def get_model():
    """Builds and compiles the CNN model architecture used for classification."""
    input_shape = (41489, 10, 1)
    inputs = tf1.keras.layers.Input(shape=input_shape, name="inputs")

    x = Conv2D(64, (5, 5), padding="same", kernel_regularizer=l2(0.001))(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid', name="outputs")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0002),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

#IG Attribution Computation
for subj_id in range(1, 87):
    file_path = os.path.join(pickle_path, f"{label}_sub-{subj_id}.pickle")
    
    if not os.path.exists(file_path):
        print(f"Pickle file not found for subject {subj_id}. Skipping.")
        continue

    print(f"\n Loading data for subject {subj_id}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    x_test = data['X']
    y_test = np.expand_dims(data['y'], axis=1)

    print(f"   X shape: {x_test.shape}, Y shape: {y_test.shape}")

    with DeepExplain(session=tf1.keras.backend.get_session()) as de:
        de.session.run(tf1.global_variables_initializer())
        de.session.run(tf1.local_variables_initializer())

        model = get_model()
        model.load_weights('model_100epoch_weights.h5')

        # Use the pre-sigmoid logit as the target for attribution
        logit_model = Model(inputs=model.input, outputs=model.layers[-1].input)

        baseline = np.zeros(x_test.shape[1:], dtype=x_test.dtype)

        attributions = de.explain(
            method='intgrad',
            T=logit_model(model.input),  # pre-sigmoid target
            X=model.input,
            xs=x_test,
            baseline=baseline,
            steps=150
        )

        print(f"IG computed. Shape: {attributions.shape}")

        # Save result
        output_path = os.path.join(save_dir, f"IG_sub-{subj_id}_{label}.pickle")
        with open(output_path, "wb") as out_f:
            pickle.dump(attributions, out_f)

        print(f"Saved IG for subject {subj_id} → {output_path}")

