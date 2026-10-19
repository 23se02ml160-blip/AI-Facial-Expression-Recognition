import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============ CONFIG ============
MODEL_PATH = "models/fer_model.h5"
TRAIN_DIR = "dataset/FERPlus/train"
TEST_DIR = "dataset/FERPlus/test"
BATCH_SIZE = 32

# ============ CHECK PATHS ============
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Train dir exists: {os.path.exists(TRAIN_DIR)}")
print(f"Test dir exists: {os.path.exists(TEST_DIR)}")
print("=" * 60)

# ============ LOAD MODEL ============
print("\n[INFO] Loading model...")
model = load_model(MODEL_PATH)
print(f"[INFO] Model input shape: {model.input_shape}")
print(f"[INFO] Model output shape: {model.output_shape}")
print(f"[INFO] Total parameters: {model.count_params():,}")

# ============ DETECT INPUT SIZE ============
input_shape = model.input_shape
img_size = (input_shape[1], input_shape[2])
print(f"\n[INFO] Using image size: {img_size}")
print(f"[INFO] Image channels: {input_shape[3]}")

# ============ DATA GENERATORS ============
# Match the preprocessing from training
if input_shape[3] == 1:
    # Grayscale (old model)
    print("[INFO] Using grayscale preprocessing")
    train_datagen = ImageDataGenerator(rescale=1./255, color_mode='grayscale')
    test_datagen = ImageDataGenerator(rescale=1./255, color_mode='grayscale')
else:
    # RGB (EfficientNetB3)
    print("[INFO] Using RGB preprocessing")
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=img_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=img_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\n[INFO] Train samples: {train_gen.samples}")
print(f"[INFO] Test samples: {test_gen.samples}")
print(f"[INFO] Classes: {train_gen.class_indices}")

# ============ EVALUATE ============
print("\n" + "=" * 60)
print("EVALUATING ON TRAINING DATA")
print("=" * 60)
train_loss, train_acc = model.evaluate(train_gen, verbose=1)

print("\n" + "=" * 60)
print("EVALUATING ON TEST DATA")
print("=" * 60)
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

# ============ RESULTS ============
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"📊 TRAINING ACCURACY:  {train_acc * 100:.2f}%")
print(f"📊 TEST ACCURACY:      {test_acc * 100:.2f}%")
print(f"📊 TRAINING LOSS:      {train_loss:.4f}")
print(f"📊 TEST LOSS:          {test_loss:.4f}")
print("=" * 60)