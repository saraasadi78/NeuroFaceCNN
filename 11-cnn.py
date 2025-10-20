import os
import gc
import shutil
import itertools
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

set_global_policy('mixed_float16')

# deterministic-ish runs (GPU kernels may still vary slightly)
tf.keras.utils.set_random_seed(42)

# GPU memory setup
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

saving_path = r"C:\Users\sara.asadi\Desktop\base"
base_path = r"D:\sarafiles\Face-Project"
pickle_file = os.path.join(base_path, "fmri_data.pickle")
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
X_data, y_data = data['X'], data['y']

print(f"X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")


# Model factory 
def create_optimized_cnn_model(input_shape):
    model = Sequential([
        Conv2D(64, (5, 3), kernel_initializer="he_normal", kernel_regularizer=l2(0.001), padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 1)),
        Dropout(0.3),
        
        Conv2D(32, (5, 3), kernel_initializer="he_normal", kernel_regularizer=l2(0.001), padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.35),
        
        Conv2D(32, (3, 3), kernel_initializer="he_normal", kernel_regularizer=l2(0.001), padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Flatten(),

        Dense(128, kernel_initializer="he_normal", kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.4),


        Dense(1, activation='sigmoid', dtype='float32')
    ])
    return model


#CV setup 
k = 4
batch_size = 20
epochs = 150
learning_rate = 0.0002

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

acc_per_fold = []
loss_per_fold = []
fold_histories = []

# Track best across folds
best_global_val_acc = -1.0
best_global_model_path = None
best_X_val, best_y_val = None, None
best_fold_index_global = None

save_dir = os.path.join(saving_path, "cv_models")
os.makedirs(save_dir, exist_ok=True)

#K-fold loop 
fold = 1
for train_index, val_index in skf.split(X_data, y_data):
    print(f"\n--- Fold {fold} ---")

    X_train, X_val = X_data[train_index], X_data[val_index]
    y_train, y_val = y_data[train_index], y_data[val_index]

    # Class weights for imbalance
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))

    # Model
    model = create_optimized_cnn_model(X_train.shape[1:])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks: checkpoint on best val_acc, early stop on val_loss, LR schedule
    fold_model_path = os.path.join(save_dir, f"fold{fold}_best.keras")
    cbs = [
        ModelCheckpoint(
            fold_model_path, monitor="val_accuracy",
            mode="max", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=15, min_lr=1e-11, verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1
    )

    # Evaluate using the best checkpoint for this fold
    best_fold_model = tf.keras.models.load_model(fold_model_path)
    scores = best_fold_model.evaluate(X_val, y_val, verbose=0)
    val_loss, val_acc = scores[0], scores[1]

    acc_per_fold.append(val_acc)
    loss_per_fold.append(val_loss)
    fold_histories.append(history)

    print(f"Fold {fold} — Best Val Loss: {val_loss:.4f} — Best Val Acc: {val_acc:.4f}")

    # Update global best
    if val_acc > best_global_val_acc:
        best_global_val_acc = val_acc
        best_global_model_path = fold_model_path
        best_X_val, best_y_val = X_val, y_val
        best_fold_index_global = fold - 1  # zero-based index for fold_histories

    tf.keras.backend.clear_session()
    gc.collect()
    fold += 1
model.summary()

#CV summary 
print("\n--- Cross-Validation Results ---")
for i in range(k):
    print(f"Fold {i+1} — Loss: {loss_per_fold[i]:.4f}, Acc: {acc_per_fold[i]:.4f}")
print(f"\nAverage Acc: {np.mean(acc_per_fold):.4f} ± {np.std(acc_per_fold):.4f}")
print(f"Average Loss: {np.mean(loss_per_fold):.4f}")

# Keep the single best model 
final_model_path = os.path.join(saving_path, "best_cv_model.keras")
shutil.copyfile(best_global_model_path, final_model_path)
print(f"\nSaved best model (val_acc={best_global_val_acc:.4f}) to: {final_model_path}")
print(f"Best fold index (1-based): {best_fold_index_global + 1}")

#Figures & metrics for BEST FOLD 
figdir = os.path.join(saving_path, "oct")
os.makedirs(figdir, exist_ok=True)

# Load best model and the corresponding history
best_model = tf.keras.models.load_model(best_global_model_path)
best_history = fold_histories[best_fold_index_global]

# Accuracy & Loss curves
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(best_history.history['accuracy'], label='Train Acc')
plt.plot(best_history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss
plt.subplot(1, 2, 2)
plt.plot(best_history.history['loss'], label='Train Loss')
plt.plot(best_history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
acc_loss_path = os.path.join(figdir, "bestfold_acc_loss.png")
plt.savefig(acc_loss_path, dpi=300)
plt.show()
print(f"Saved accuracy/loss figure to: {acc_loss_path}")

# ROC curve + AUC on best fold's validation set
y_val_prob = best_model.predict(best_X_val, verbose=0).ravel()
auc = roc_auc_score(best_y_val, y_val_prob)
fpr, tpr, _ = roc_curve(best_y_val, y_val_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
roc_path = os.path.join(figdir, "bestfold_roc.png")
plt.savefig(roc_path, dpi=300)
plt.show()
print(f"Saved ROC figure to: {roc_path}")
print(f"Validation AUC (best fold): {auc:.4f}")

# Confusion Matrix on best fold's validation set (thr=0.5)
y_val_pred = (y_val_prob >= 0.5).astype(np.int32)
cm = confusion_matrix(best_y_val, y_val_pred)

plt.figure(figsize=(5, 5))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'])
plt.yticks(tick_marks, ['Class 0', 'Class 1'])

thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
cm_path = os.path.join(figdir, "bestfold_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()
print(f"Saved confusion matrix figure to: {cm_path}")

# Classification report for context
print("\nClassification Report (best fold val set, thr=0.5):")
print(classification_report(best_y_val, y_val_pred, digits=3))


final_model_h5 = os.path.join(saving_path, "model83-new.h5")
final_weights_h5 = os.path.join(saving_path, "model83_weights-new.h5")

best_model.save(final_model_h5)              # full model in HDF5
best_model.save_weights(final_weights_h5)    # weights-only in HDF5

print(f"Saved full HDF5 model to: {final_model_h5}")
print(f"Saved HDF5 weights to:    {final_weights_h5}")


# y_true: true binary labels (best_y_val)
# y_probs: predicted probabilities (best_model.predict(best_X_val).ravel())
# Use same validation set you used for AUC and confusion matrix

prob_true, prob_pred = calibration_curve(best_y_val, y_val_prob, n_bins=10)

plt.figure(figsize=(7, 7))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='CNN')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Reliability Diagram / Calibration Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figdir, "reliability_diagram.png"), dpi=300)
plt.show()
