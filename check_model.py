import tensorflow as tf
import os

print("=" * 60)
print("MODEL DIAGNOSTIC")
print("=" * 60)

# Check both models
for model_file in ['models/fer_model.h5', 'models/ferplus_model.keras']:
    if os.path.exists(model_file):
        print(f"\n📦 {model_file}")
        print(f"   Size: {os.path.getsize(model_file) / (1024*1024):.2f} MB")
        try:
            model = tf.keras.models.load_model(model_file)
            print(f"   Input shape:  {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Parameters:   {model.count_params():,}")
            
            # Show first few layers
            print(f"   First layer:  {model.layers[0].name}")
            print(f"   Last layer:   {model.layers[-1].name}")
        except Exception as e:
            print(f"   ❌ Load failed: {e}")
    else:
        print(f"\n❌ Not found: {model_file}")

print("\n" + "=" * 60)