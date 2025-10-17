import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# --- Đường dẫn file CSV từ video ---
csv_path = "dataset/keypoints_video.csv"

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)

# Bỏ các hàng không có nhãn (nếu có)
df = df[df['label'].notnull() & (df['label'] != "unknown")]

# --- Tách đặc trưng & nhãn ---
# Ở CSV video, cột 'frame' thay cho 'filename'
X = df.drop(columns=[ "label"]).values
y = df["label"].values

# --- Encode nhãn thành số ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Chia tập train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Huấn luyện model ---
print("🚀 Training RandomForestClassifier from video frames...")
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- Đánh giá ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc*100:.2f}%\n")
print("📊 Classification report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- Confusion matrix ---
print("\n🧩 Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Lưu model & encoder ---
joblib.dump(model, "pose_classifier_video.pkl")
joblib.dump(label_encoder, "label_encoder_video.pkl")
print("\n✅ Model saved: pose_classifier_video.pkl")
print("✅ Label encoder saved: label_encoder_video.pkl")
