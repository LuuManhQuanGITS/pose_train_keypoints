import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- Load dataset ---
csv_path = "dataset/keypoints_video.csv"
df = pd.read_csv(csv_path)

print(f"📊 Tổng số mẫu: {len(df)}")
print(f"📋 Các nhãn: {df['label'].unique()}")
print(f"📈 Phân bố nhãn:\n{df['label'].value_counts()}")

# --- Tách features và labels ---
X = df.drop("label", axis=1).values
y = df["label"].values

# --- Encode labels ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"\n🏷️ Số lượng classes: {num_classes}")
print(f"🏷️ Classes: {label_encoder.classes_}")

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📦 Train set: {X_train.shape[0]} samples")
print(f"📦 Test set: {X_test.shape[0]} samples")

# --- Build Dense Neural Network ---
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    
    # Dense layers với Batch Normalization và Dropout
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(num_classes, activation='softmax')
])

# --- Compile model ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Callbacks ---
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# --- Train model ---
print("\n🚀 Bắt đầu training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10000,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# --- Plot training history ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("\n📊 Biểu đồ đã lưu: training_history.png")

# --- Save Keras model ---
model.save('pose_classifier_dense.h5')
print("\n💾 Model Keras đã lưu: pose_classifier_dense.h5")

# --- Convert to TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Tùy chọn tối ưu hóa (có thể bỏ comment để giảm kích thước)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save TFLite model
tflite_path = 'pose_classifier_dense.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ TFLite model đã lưu: {tflite_path}")
print(f"📦 Kích thước: {len(tflite_model) / 1024:.2f} KB")

# --- Save Label Encoder ---
import joblib
joblib.dump(label_encoder, 'label_encoder_dense.pkl')
print("✅ Label encoder đã lưu: label_encoder_dense.pkl")

# --- Test TFLite model ---
print("\n🧪 Testing TFLite model...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test với một sample
test_sample = X_test[0:1].astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_sample)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])
tflite_pred = np.argmax(tflite_output)

# So sánh với model gốc
keras_pred = np.argmax(model.predict(test_sample, verbose=0))

print(f"Keras prediction: {label_encoder.classes_[keras_pred]}")
print(f"TFLite prediction: {label_encoder.classes_[tflite_pred]}")
print(f"Match: {keras_pred == tflite_pred}")

print("\n✅ Hoàn tất training và export TFLite!")