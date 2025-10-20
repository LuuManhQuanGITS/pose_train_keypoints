import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="pose_classifier_dense.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Load Label Encoder ---
label_encoder = joblib.load("label_encoder_dense.pkl")

# --- MediaPipe setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, 
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# --- Normalize keypoints ---
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
    return np.array(norm_points, dtype=np.float32).reshape(1, -1)

# --- Predict với TFLite ---
def predict_tflite(keypoints):
    interpreter.set_tensor(input_details[0]['index'], keypoints)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(output[0])
    confidence = output[0][pred_idx]
    pred_label = label_encoder.classes_[pred_idx]
    return pred_label, confidence

# --- Video/Camera ---
video_path = "1051956901-preview.mp4"  # hoặc 0 cho webcam
cap = cv2.VideoCapture(video_path)

print("🎥 Nhấn 'q' để thoát")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)
    
    if result.pose_landmarks:
        # Vẽ skeleton
        mp_drawing.draw_landmarks(
            frame, 
            result.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )
        
        # Predict
        keypoints = normalize_keypoints(result.pose_landmarks.landmark)
        pred_label, confidence = predict_tflite(keypoints)
        
        # Hiển thị kết quả
        if confidence > 0.8:
            text = f"{pred_label}: {confidence*100:.1f}%"
            cv2.putText(frame, text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "No pose detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Resize để hiển thị
    height, width = frame.shape[:2]
    max_height = 1000
    if height > max_height:
        scale = max_height / height
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    
    cv2.imshow("Pose Classification - TFLite", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("✅ Đã đóng chương trình")