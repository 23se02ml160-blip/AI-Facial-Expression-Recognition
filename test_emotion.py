import numpy as np
import cv2
import tensorflow as tf
from utils import EMOTIONS, preprocess_face

print("=" * 60)
print("EMOTION MODEL TEST")
print("=" * 60)

# Load model
model = tf.keras.models.load_model("models/fer_model.h5")
print(f"Model input shape:  {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"EMOTIONS list ({len(EMOTIONS)}): {EMOTIONS}")

if model.output_shape[-1] != len(EMOTIONS):
    print(f"\n❌ MISMATCH! Model outputs {model.output_shape[-1]} classes but EMOTIONS has {len(EMOTIONS)}")
    print("   This is why you only see Neutral — fix the EMOTIONS list.")
    exit()

# Create a synthetic face (random image)
test_face = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

# Preprocess
processed = preprocess_face(test_face)
print(f"\nPreprocessed shape: {processed.shape}")

if processed.shape != (1, 224, 224, 3):
    print(f"❌ WRONG SHAPE! Expected (1, 224, 224, 3), got {processed.shape}")
    exit()

# Predict
preds = model.predict(processed, verbose=0)
print(f"\nPrediction probabilities:")
for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds[0])):
    bar = "█" * int(prob * 50)
    print(f"  {emotion:12s}: {prob*100:6.2f}%  {bar}")

idx = int(np.argmax(preds))
print(f"\nPredicted: {EMOTIONS[idx]} ({preds[0][idx]*100:.2f}%)")