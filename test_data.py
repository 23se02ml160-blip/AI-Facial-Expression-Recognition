import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = "dataset/FERPlus"
TRAIN_DIR = os.path.join(BASE_DIR, "train")

datagen = ImageDataGenerator()
gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

print(f"Found {gen.samples} images, {len(gen.class_indices)} classes")
print("Class indices:", gen.class_indices)

# Try to load one batch
x, y = next(gen)
print(f"Batch shape: {x.shape}, labels shape: {y.shape}")
print("Pixel values range:", x.min(), "to", x.max())