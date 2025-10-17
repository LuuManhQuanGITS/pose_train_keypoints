import cv2
import mediapipe as mp
import numpy as np
import joblib

# --- Load model & encoder ---
model = joblib.load("pose_classifier_video.pkl")
label_encoder = joblib.load("label_encoder_video.pkl")
print("✅ Model & encoder loaded successfully!\n")

# --- Mediapipe setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Hàm normalize keypoints ---
def normalize_keypoints(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    norm_points = []
    for lm in landmarks:
        norm_x = (lm.x - min_x) / width if width > 0 else 0
        norm_y = (lm.y - min_y) / height if height > 0 else 0
        norm_points.extend([norm_x, norm_y])
    return np.array(norm_points).reshape(1, -1)

# --- Video capture ---
video_path = "Screen_Recording_20250910_133430.mp4"  # hoặc 0 để dùng webcam
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        keypoints_norm = normalize_keypoints(result.pose_landmarks.landmark)

        # Dự đoán hành động
        pred_encoded = model.predict(keypoints_norm)
        pred_label = label_encoder.inverse_transform(pred_encoded)[0]

        # Vẽ pose và nhãn dự đoán lên frame
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(
            frame, f"Predicted: {pred_label}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2
        )
    
    
    
    
    # Giả sử frame có kích thước gốc
    height, width = frame.shape[:2]
    
    # Giới hạn chiều cao tối đa là 1000
    max_height = 1000
    
    if height > max_height:
        # Tính tỉ lệ co
        scale = max_height / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame_resized = cv2.resize(frame, (new_width, new_height))
    else:
        frame_resized = frame    

    cv2.imshow("Pose Prediction Video", frame_resized)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
