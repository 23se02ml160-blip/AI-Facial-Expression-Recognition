import os
import tensorflow as tf

MODELS = [
    "models/ferplus_model.keras",
    "models/fer_model.h5",
    "models/fer_model_fast.h5",
    "models/fer_model_local.h5",
]

print("=" * 70)
print("CHECKING ALL MODELS")
print("=" * 70)

for path in MODELS:
    if not os.path.exists(path):
        print(f"\n❌ {path} — NOT FOUND")
        continue

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n📦 {path} ({size_mb:.2f} MB)")
    try:
        model = tf.keras.models.load_model(path)
        print(f"   Input shape:  {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Classes:      {model.output_shape[-1]}")
        print(f"   Parameters:   {model.count_params():,}")

        if model.output_shape[-1] == 8:
            print(f"   ✅ THIS IS THE 8-CLASS MODEL (with Contempt)!")
        else:
            print(f"   ⚠️  This is a {model.output_shape[-1]}-class model")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")

print("\n" + "=" * 70)