import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

print("🔄 Loading models...")
label_encoder = joblib.load("label_encoder_video.pkl")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="pose_classifier_video.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Output shape: {output_details[0]['shape']}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

print("✅ Starting video processing...\n")
video_path = "3840704143-preview.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    
    if result.pose_landmarks:
        # Normalize keypoints
        keypoints_norm = normalize_keypoints(result.pose_landmarks.landmark)
        
        # TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], keypoints_norm)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get predicted class
        if len(output_data.shape) == 1:
            pred_index = int(output_data[0])
        else:
            pred_index = int(output_data[0][0])
        
        pred_label = label_encoder.classes_[pred_index]
        
        # Draw pose and prediction
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        cv2.putText(
            frame,
            f"Frame: {frame_count} | Predicted: {pred_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    
    # Resize for display
    height, width = frame.shape[:2]
    if height > 1000:
        scale = 1000 / height
        new_width = int(width * scale)
        new_height = 1000
        frame = cv2.resize(frame, (new_width, new_height))
    
    cv2.imshow("TFLite Pose Prediction", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\n✅ Processed {frame_count} frames")