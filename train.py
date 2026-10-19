# pyright: reportGeneralTypeIssues=false, reportMissingImports=false
"""
Local Training Script — Optimized for CPU
Trains MobileNetV2 on FER+ dataset in ~1-2 hours
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from collections import Counter

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_P1 = 6
EPOCHS_P2 = 12

TRAIN_DIR = "dataset/FERPlus/train"
TEST_DIR = "dataset/FERPlus/test"
MODEL_SAVE = "models/fer_model.h5"
CLASS_INDICES_SAVE = "models/class_indices.json"

os.makedirs("models", exist_ok=True)

print("=" * 60)
print("LOCAL TRAINING — CPU OPTIMIZED")
print("=" * 60)
print(f"TensorFlow: {tf.__version__}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print("=" * 60)

if not os.path.exists(TRAIN_DIR):
    print(f"[ERROR] Not found: {TRAIN_DIR}")
    raise SystemExit

print("\n[INFO] Loading data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical'
)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

print(f"\n[OK] Classes ({train_gen.num_classes}): {train_gen.class_indices}")
print(f"[OK] Train: {train_gen.samples} | Test: {test_gen.samples}")

idx_map = {str(i): c for c, i in train_gen.class_indices.items()}
with open(CLASS_INDICES_SAVE, "w", encoding="utf-8") as f:
    json.dump(idx_map, f, indent=2)
print(f"[INFO] Saved {CLASS_INDICES_SAVE}: {idx_map}")

counts = Counter(train_gen.classes)
total = sum(counts.values())
class_weights = {i: total / (train_gen.num_classes * counts.get(i, 1))
                 for i in range(train_gen.num_classes)}

print("\n" + "=" * 60)
print("BUILDING MOBILENETV2 MODEL")
print("=" * 60)

base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(train_gen.num_classes, activation='softmax', dtype='float32')(x)
model = Model(inputs=base.input, outputs=out)

print(f"[OK] Total params: {model.count_params():,}")
print(f"[OK] Trainable params: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

print("\n" + "=" * 60)
print("PHASE 1: Frozen Base Training")
print("=" * 60)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb1 = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy', save_best_only=True, verbose=1)
]

h1 = model.fit(train_gen, epochs=EPOCHS_P1, validation_data=test_gen,
               class_weight=class_weights, callbacks=cb1, verbose=1)

print(f"\n[OK] Phase 1 best val_accuracy: {max(h1.history['val_accuracy']) * 100:.2f}%")

print("\n" + "=" * 60)
print("PHASE 2: Fine-Tuning")
print("=" * 60)

base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb2 = [
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy', save_best_only=True, verbose=1)
]

h2 = model.fit(train_gen, epochs=EPOCHS_P2, validation_data=test_gen,
               class_weight=class_weights, callbacks=cb2, verbose=1)

print("\n" + "=" * 60)
print("PER-CLASS ACCURACY ANALYSIS")
print("=" * 60)

best_model = tf.keras.models.load_model(MODEL_SAVE)

test_gen.reset()
predictions = best_model.predict(test_gen, verbose=1)
pred_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes

idx_to_class = {v: k for k, v in test_gen.class_indices.items()}

print("\nPer-Emotion Accuracy:")
for i in range(len(idx_to_class)):
    mask = true_classes == i
    if mask.sum() > 0:
        acc = (pred_classes[mask] == true_classes[mask]).mean() * 100
        emotion = idx_to_class[i].capitalize()
        bar = "#" * int(acc / 5)
        print(f"  {emotion:12s}: {acc:6.2f}%  {bar}")

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

_, train_acc = best_model.evaluate(train_gen, verbose=0)
_, test_acc = best_model.evaluate(test_gen, verbose=0)

print(f"TRAINING ACCURACY:  {train_acc * 100:.2f}%")
print(f"TEST ACCURACY:      {test_acc * 100:.2f}%")
print(f"Phase 1 Best Val:   {max(h1.history['val_accuracy']) * 100:.2f}%")
print(f"Phase 2 Best Val:   {max(h2.history['val_accuracy']) * 100:.2f}%")
print("=" * 60)

idx_to_class_final = {v: k for k, v in train_gen.class_indices.items()}
EMOTIONS = [idx_to_class_final[i] for i in range(train_gen.num_classes)]
print(f"\nEMOTIONS list for utils.py:")
print(f"   {EMOTIONS}")

print(f"\nModel saved:   {MODEL_SAVE}")
print(f"Class map:     {CLASS_INDICES_SAVE}")
print(f"Size:          {os.path.getsize(MODEL_SAVE) / (1024*1024):.2f} MB")