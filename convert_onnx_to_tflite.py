import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

print("🔄 Converting ONNX to TFLite...")

# Step 1: Load ONNX model
print("📁 Loading ONNX model...")
onnx_model = onnx.load("pose_classifier.onnx")

# Step 2: Convert ONNX to TensorFlow
print("🔄 Converting to TensorFlow format...")
tf_rep = prepare(onnx_model)

# Step 3: Export to SavedModel format
print("💾 Exporting to SavedModel...")
tf_rep.export_graph("pose_classifier_tf")

# Step 4: Convert to TFLite
print("🔄 Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("pose_classifier_tf")

# Optional: Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Step 5: Save TFLite model
output_file = "pose_classifier.tflite"
with open(output_file, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved as '{output_file}'")

# Step 6: Verify the model
print("\n📊 Model Information:")
interpreter = tf.lite.Interpreter(model_path=output_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Input dtype: {input_details[0]['dtype']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

print("\n✅ Conversion complete!")