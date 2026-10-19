import numpy as np
import tensorflow as tf
from utils import EMOTIONS, preprocess_face

print("=" * 60)
print("EMOTION MODEL FINAL TEST")
print("=" * 60)

# Load model
model = tf.keras.models.load_model("models/fer_model.h5")
print(f"\nModel input shape:  {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Model classes:      {model.output_shape[-1]}")

print(f"\nEMOTIONS list ({len(EMOTIONS)} items):")
print(f"   {EMOTIONS}")

if model.output_shape[-1] != len(EMOTIONS):
    print(f"\n❌ MISMATCH!")
    print(f"   Model outputs {model.output_shape[-1]} classes")
    print(f"   EMOTIONS has {len(EMOTIONS)} items")
    print(f"\n   Fix: Set EMOTIONS to match the model.")
    exit()

print("\n✅ Model classes match EMOTIONS list")

# Test with random image
test_face = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
processed = preprocess_face(test_face)
print(f"\nPreprocessed shape: {processed.shape}")

if processed.shape != (1, 224, 224, 3):
    print(f"❌ WRONG SHAPE — expected (1, 224, 224, 3)")
    exit()

# Predict
preds = model.predict(processed, verbose=0)
print(f"\nPrediction probabilities:")
for emotion, prob in zip(EMOTIONS, preds[0]):
    bar = "█" * int(prob * 40)
    print(f"  {emotion:12s}: {prob*100:6.2f}%  {bar}")

idx = int(np.argmax(preds))
print(f"\n✅ Predicted: {EMOTIONS[idx]} ({preds[0][idx]*100:.2f}%)")
print("=" * 60)