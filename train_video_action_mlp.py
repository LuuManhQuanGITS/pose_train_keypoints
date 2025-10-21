import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import joblib

# --- Load dataset ---
csv_path = "dataset/keypoints_image_series.csv"
df = pd.read_csv(csv_path)

print(f"📊 Tổng số series: {len(df)}")
print(f"📋 Các nhãn: {df['label'].unique()}")
print(f"📈 Phân bố nhãn:\n{df['label'].value_counts()}")

# --- Tách features và labels ---
y = df['label'].astype(str).values

X = df.drop(['series_id', 'label'], axis=1).values.astype(np.float32)

# --- Chuẩn hóa object centers và cls_id ---
# Giả sử object centers + cls_id là các cột cuối: last num_objects*3 cột
num_objects = 5  # đúng với số object trong CSV
num_object_cols = num_objects * 3  # x,y,cls_id
object_data = X[:, -num_object_cols:]

# Normalize x,y (đã là 0~1) -> giữ nguyên
# Normalize cls_id: chia cho max cls_id để scale 0~1
cls_ids = object_data[:, 2::3]
max_cls_id = np.max(cls_ids)
cls_ids = np.where(cls_ids < 0, 0, cls_ids)  # đổi -1 padding thành 0
cls_ids = cls_ids / max_cls_id if max_cls_id > 0 else cls_ids
object_data[:, 2::3] = cls_ids

# Gán lại X
X[:, -num_object_cols:] = object_data

# --- Encode labels ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"\n🏷️ Số lượng classes: {num_classes}")
print(f"🏷️ Classes: {label_encoder.classes_}")

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"\n📦 Train set: {X_train.shape[0]} samples")
print(f"📦 Test set: {X_test.shape[0]} samples")

# --- Build Dense Neural Network ---
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
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
    monitor='val_loss', patience=15, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
)

# --- Train model ---
print("\n🚀 Bắt đầu training series frame...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=1000,
    batch_size=16,
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
plt.savefig('training_history_series.png')
print("\n📊 Biểu đồ đã lưu: training_history_series.png")

# --- Save Keras model ---
model.save('pose_classifier_dense_series.h5')
print("\n💾 Model Keras đã lưu: pose_classifier_dense_series.h5")

# --- Convert to TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = 'pose_classifier_dense_series.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"✅ TFLite model đã lưu: {tflite_path} ({len(tflite_model)/1024:.2f} KB)")

# --- Save Label Encoder ---
joblib.dump(label_encoder, 'label_encoder_dense_series.pkl')
print("✅ Label encoder đã lưu: label_encoder_dense_series.pkl")
