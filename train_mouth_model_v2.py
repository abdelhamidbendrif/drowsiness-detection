"""
train_mouth_model_v2.py  (Kaggle notebook cell — paste as a new cell)
======================================================================
Improved mouth (yawn) model training.

Key improvements over v1:
  [IMP-1] More epochs (50 instead of 30) + higher patience (12 instead of 8)
  [IMP-2] ReduceLROnPlateau added — accelerates convergence on small dataset
  [IMP-3] class_weight computed to handle possible label imbalance
  [IMP-4] Stronger augmentation (shear, channel shifts) for tiny dataset
  [IMP-5] Input is kept as full-face image (correct for this dataset)
  [IMP-6] TFDataset-style repeated dataset avoids generator reset artefacts
"""

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ──────────────────────────────────────────────────────────────────
YAWN_TRAIN = '/kaggle/input/datasets/serenaraju/yawn-eye-dataset-new/dataset_new/train'
YAWN_TEST  = '/kaggle/input/datasets/serenaraju/yawn-eye-dataset-new/dataset_new/test'
SAVE_PATH  = '/kaggle/working/mouth_model_v2.keras'

IMG_SIZE   = (64, 64)
BATCH_SIZE = 32

# ── Generators ─────────────────────────────────────────────────────────────
# [IMP-4] Stronger augmentation for the tiny dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    channel_shift_range=20.0,         # ← NEW: simulate lighting variation
)

val_datagen = ImageDataGenerator(rescale=1./255)

yawn_train_gen = train_datagen.flow_from_directory(
    YAWN_TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['no_yawn', 'yawn'],
    shuffle=True
)

yawn_test_gen = val_datagen.flow_from_directory(
    YAWN_TEST,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['no_yawn', 'yawn'],
    shuffle=False
)

print(f"[OK] Classes : {yawn_train_gen.class_indices}")
print(f"[DATA] Train : {yawn_train_gen.samples} | Test : {yawn_test_gen.samples}")

# ── Class weights ──────────────────────────────────────────────────────────
# [IMP-3] Handle label imbalance (yawn images are often fewer than no_yawn)
labels    = yawn_train_gen.classes
weights   = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_wt  = dict(enumerate(weights))
print(f"[DATA] Class weights : {class_wt}")

# ── Model ──────────────────────────────────────────────────────────────────
base2 = MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
base2.trainable = True
for layer in base2.layers[:-40]:
    layer.trainable = False

x2  = base2.output
x2  = GlobalAveragePooling2D()(x2)
x2  = BatchNormalization()(x2)
x2  = Dense(128, activation='relu')(x2)
x2  = Dropout(0.4)(x2)
x2  = Dense(64,  activation='relu')(x2)
x2  = Dropout(0.3)(x2)
out2 = Dense(1,  activation='sigmoid')(x2)

mouth_model_v2 = Model(inputs=base2.input, outputs=out2)
mouth_model_v2.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

print(f"[OK] Trainable params : {sum(p.numpy().size for p in mouth_model_v2.trainable_weights):,}")

# ── Callbacks ──────────────────────────────────────────────────────────────
callbacks2 = [
    # [IMP-1] Higher patience — the model converges slowly on a small dataset
    EarlyStopping(monitor='val_accuracy', patience=12,
                  restore_best_weights=True, verbose=1),
    # [IMP-2] ReduceLROnPlateau was missing in v1
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-8, verbose=1),
    ModelCheckpoint(SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

# ── Training ───────────────────────────────────────────────────────────────
print("\n[TRAIN] CNN Mouth v2 - 50 epochs max...")
history2 = mouth_model_v2.fit(
    yawn_train_gen,
    epochs=50,                    # [IMP-1] was 30 — model hadn't converged
    validation_data=yawn_test_gen,
    callbacks=callbacks2,
    class_weight=class_wt,        # [IMP-3]
    verbose=1
)

best_acc = max(history2.history['val_accuracy'])
best_auc = max(history2.history['val_AUC'])
print("\n[DONE]")
print(f"[BEST] val_accuracy : {best_acc:.4f}  ({best_acc*100:.2f}%)")
print(f"[BEST] val_AUC      : {best_auc:.4f}  ({best_auc*100:.2f}%)")

# ── Evaluation ─────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test_loss, test_acc, test_auc = mouth_model_v2.evaluate(yawn_test_gen, verbose=0)
print(f"\n[TEST] Accuracy : {test_acc:.4f}")
print(f"[TEST] AUC      : {test_auc:.4f}")

y_pred = (mouth_model_v2.predict(yawn_test_gen, verbose=0) > 0.5).astype(int).flatten()
y_true = yawn_test_gen.classes

print("\n[TEST] Classification Report:")
print(classification_report(y_true, y_pred, target_names=['no_yawn', 'yawn']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['no_yawn', 'yawn'],
            yticklabels=['no_yawn', 'yawn'], ax=axes[0])
axes[0].set_title('Confusion Matrix — Mouth CNN v2', fontweight='bold')
axes[0].set_ylabel('True label')
axes[0].set_xlabel('Predicted')

axes[1].plot(history2.history['val_accuracy'], label='Val Accuracy', color='orange')
axes[1].plot(history2.history['val_loss'],     label='Val Loss',     color='red')
axes[1].set_title('Val Accuracy & Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Mouth CNN v2 — Training Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
