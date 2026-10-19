import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = "dataset/FERPlus"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")
if not os.path.exists(VAL_DIR):
    VAL_DIR = os.path.join(BASE_DIR, "test")

print("Train folder:", TRAIN_DIR)
print("Validation folder:", VAL_DIR)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print("Train samples:", train_gen.samples)
print("Val samples:", val_gen.samples)
print("Class indices:", train_gen.class_indices)