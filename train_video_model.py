import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# --- Đường dẫn file CSV từ video ---
csv_path = "dataset/keypoints_video.csv"

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)

# Bỏ các hàng không có nhãn (nếu có)
df = df[df["label"].notnull() & (df["label"] != "unknown")]

# --- Tách đặc trưng & nhãn ---
X = df.drop(columns=["label"]).values
y = df["label"].values

# --- Encode nhãn thành số ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Chuyển sang one-hot encoding cho neural network
y_categorical = keras.utils.to_categorical(y_encoded, num_classes)

# --- Chia tập train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Chuẩn hóa dữ liệu
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Xây dựng model Neural Network ---
print("🚀 Building Neural Network model...")
input_dim = X_train.shape[1]

model = keras.Sequential(
    [
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# --- Compile model ---
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

# --- Huấn luyện model ---
print("\n🚀 Training model...")
history = model.fit(
    X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1
)

# --- Đánh giá ---
print("\n📊 Evaluating model...")
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"🎯 Test Accuracy: {test_acc*100:.2f}%")

# Dự đoán và classification report
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print("\n📊 Classification report:")
print(
    classification_report(
        y_test_classes, y_pred_classes, target_names=label_encoder.classes_
    )
)

print("\n🧩 Confusion matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes))

# --- Lưu model Keras ---
model.save("pose_classifier_video.h5")
print("\n✅ Keras model saved: pose_classifier_video.h5")

# --- Chuyển đổi sang TFLite ---
print("\n🔄 Converting to TFLite...")

# Tạo TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Tùy chọn tối ưu hóa (optional)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Chuyển đổi
tflite_model = converter.convert()

# Lưu file TFLite
tflite_path = "pose_classifier_video.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved: {tflite_path}")

# --- Lưu label encoder và scaler ---
joblib.dump(label_encoder, "label_encoder_video.pkl")
joblib.dump(scaler, "scaler_video.pkl")
print("✅ Label encoder saved: label_encoder_video.pkl")
print("✅ Scaler saved: scaler_video.pkl")

# --- Test TFLite model ---
print("\n🧪 Testing TFLite model...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test với một mẫu
test_sample = X_test_scaled[0:1].astype(np.float32)
interpreter.set_tensor(input_details[0]["index"], test_sample)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]["index"])

print(f"Original prediction: {label_encoder.classes_[y_test_classes[0]]}")
print(f"TFLite prediction: {label_encoder.classes_[np.argmax(tflite_output)]}")
print("\n✅ Training and conversion completed!")
