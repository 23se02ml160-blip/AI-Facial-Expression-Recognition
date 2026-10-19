import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 12

TRAIN_DIR = "dataset/FERPlus/train"
TEST_DIR = "dataset/FERPlus/test"
MODEL_SAVE = "models/fer_model.h5"
CLASS_INDICES_SAVE = "models/class_indices.json"

os.makedirs("models", exist_ok=True)

print("=" * 60)
print("FAST RETRAIN — MobileNetV2 @ 96x96")
print("=" * 60)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
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

print(f"Classes: {train_gen.class_indices}")
print(f"Train: {train_gen.samples} | Test: {test_gen.samples}")

idx_map = {str(i): c for c, i in train_gen.class_indices.items()}
with open(CLASS_INDICES_SAVE, "w", encoding="utf-8") as f:
    json.dump(idx_map, f, indent=2)
print(f"[INFO] Saved {CLASS_INDICES_SAVE}")

from collections import Counter
counts = Counter(train_gen.classes)
total = sum(counts.values())
class_weights = {i: total / (train_gen.num_classes * counts.get(i, 1))
                 for i in range(train_gen.num_classes)}

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
out = Dense(train_gen.num_classes, activation='softmax', dtype='float32')(x)
model = Model(inputs=base.input, outputs=out)

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\nTotal params: {model.count_params():,}")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(train_gen, epochs=EPOCHS, validation_data=test_gen,
                    class_weight=class_weights, callbacks=callbacks)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
train_loss, train_acc = model.evaluate(train_gen, verbose=1)
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"TRAINING ACCURACY:  {train_acc * 100:.2f}%")
print(f"TEST ACCURACY:      {test_acc * 100:.2f}%")
print(f"\nModel saved to: {MODEL_SAVE}")
print(f"Class map:      {CLASS_INDICES_SAVE}")
print("=" * 60)