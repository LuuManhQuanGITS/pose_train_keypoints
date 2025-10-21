import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from collections import deque
from ultralytics import YOLO
import uuid

LABELS = {
    0:"UNKNOWN",
    1: "HANDS_ABOVE_HEAD",
    2: "BENDING_TWISTING_NECK_BACK",
    3: "SQUATTING_OR_KNEELING",
    4: "USING_FINGERS",
    5: "ONE_HAND_LIFT_HEAVY",
    6: "BENDING_DOWN_LIFT_HEAVY",
    7: "LIFT_HEAVY_BOTH_HANDS",
    8: "LIFT_HEAVY_SHOULDERS_BACK",
    9: "STRIKING_WITH_HAND_OR_KNEE"
}

video_path = "14547976_3840_2160_60fps.mp4"
output_csv = "dataset/keypoints_video_series.csv"
window_size = 50

# Cải thiện cấu hình MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Tăng độ tin cậy và sử dụng model_complexity cao hơn
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,  # 0, 1, hoặc 2 (cao nhất)
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.3,  # Giảm để dễ detect hơn
    min_tracking_confidence=0.3
)

yolo_model = YOLO("yolo11m.pt")

num_pose = 33
num_hand = 21
num_objects = 5
total_kp = num_pose + 2*num_hand + num_objects
seq_len = window_size
header = ["series_id","label"] + [f"x{i}_t{t}" for t in range(seq_len) for i in range(total_kp)] + \
         [f"y{i}_t{t}" for t in range(seq_len) for i in range(total_kp)]
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
if not os.path.exists(output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def normalize_landmarks(landmarks, num_expected):
    """Normalize landmarks với số lượng cố định"""
    if not landmarks or len(landmarks) == 0:
        return [0]*(num_expected * 2)
    
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x if max_x - min_x > 1e-6 else 1
    height = max_y - min_y if max_y - min_y > 1e-6 else 1
    
    norm = []
    for lm in landmarks:
        norm.append((lm.x - min_x)/width)
        norm.append((lm.y - min_y)/height)
    return norm

seq_buffer = deque(maxlen=window_size)
cap = cv2.VideoCapture(video_path)
frame_idx = 0

# Thêm counter để theo dõi detection
pose_detected = 0
left_hand_detected = 0
right_hand_detected = 0

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing frame để cải thiện detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tăng độ sáng nếu ảnh tối
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if np.mean(v) < 100:  # Nếu ảnh tối
            v = cv2.add(v, 30)
            v = np.clip(v, 0, 255)
            hsv = cv2.merge([h, s, v])
            frame_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
        
        # Process với MediaPipe
        result = holistic.process(frame_rgb)

        # Vẽ landmarks với style đẹp hơn
        if result.pose_landmarks:
            pose_detected += 1
            mp_drawing.draw_landmarks(
                frame, 
                result.pose_landmarks, 
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if result.left_hand_landmarks:
            left_hand_detected += 1
            mp_drawing.draw_landmarks(
                frame, 
                result.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            # Vẽ text thông báo
            cv2.putText(frame, "LEFT HAND DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if result.right_hand_landmarks:
            right_hand_detected += 1
            mp_drawing.draw_landmarks(
                frame, 
                result.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            cv2.putText(frame, "RIGHT HAND DETECTED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # YOLO detection
        yolo_results = yolo_model.predict(frame_rgb, verbose=False)
        obj_landmarks = []
        obj_count = 0
        if yolo_results and len(yolo_results[0].boxes) > 0:
            for box in yolo_results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = yolo_model.names[cls_id]
                
                if class_name.lower() == "person":
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                # Vẽ bounding box cho object
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                
                # Vẽ tên class và confidence
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                            (int(x1) + label_size[0], int(y1)), (0, 255, 255), -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Vẽ center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Lưu normalized coordinates
                cx_norm = cx / frame.shape[1]
                cy_norm = cy / frame.shape[0]
                obj_landmarks.append((cx_norm, cy_norm))
                obj_count += 1
                
                if len(obj_landmarks) >= num_objects:
                    break
        
        while len(obj_landmarks) < num_objects:
            obj_landmarks.append((0, 0))

        # Tạo keypoint vector
        kp_vector = []
        kp_vector += normalize_landmarks(
            result.pose_landmarks.landmark if result.pose_landmarks else [], 
            num_pose
        )
        kp_vector += normalize_landmarks(
            result.left_hand_landmarks.landmark if result.left_hand_landmarks else [], 
            num_hand
        )
        kp_vector += normalize_landmarks(
            result.right_hand_landmarks.landmark if result.right_hand_landmarks else [], 
            num_hand
        )
        for cx, cy in obj_landmarks:
            kp_vector.append(cx)
            kp_vector.append(cy)

        seq_buffer.append(kp_vector)

        # Hiển thị thông tin detection
        info_text = f"Frame: {frame_idx} | Pose: {pose_detected} | L.Hand: {left_hand_detected} | R.Hand: {right_hand_detected} | Objects: {obj_count}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Hiển thị số frame trong buffer
        cv2.putText(frame, f"Buffer: {len(seq_buffer)}/{window_size}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Resize và hiển thị
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        max_height = 1000
        if height > max_height:
            scale = max_height / height
            display_frame = cv2.resize(display_frame, (int(width*scale), int(height*scale)))
        
        cv2.imshow("Label Series Frame", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Labeling khi đủ window_size
        if len(seq_buffer) == window_size:
            series_uuid = str(uuid.uuid4())
            print(f"\n{'='*60}")
            print(f"Series {series_uuid} ending at frame {frame_idx}")
            print(f"Detection stats: Pose={pose_detected}, Left Hand={left_hand_detected}, Right Hand={right_hand_detected}")
            print(f"{'='*60}")
            for k, v in LABELS.items():
                print(f"{k}: {v}")
            print("Nhập số (1–9) để gán nhãn series, '0' để bỏ qua, 'ESC' để thoát.")

            while True:
                key = cv2.waitKey(0)
                if key == 27:  # ESC
                    cap.release()
                    cv2.destroyAllWindows()
                    holistic.close()
                    print("\n✅ Hoàn tất extract series frame với series_id dạng UUID.")
                    exit()
                elif chr(key).isdigit() and int(chr(key)) in LABELS:
                    label = LABELS[int(chr(key))]
                    seq_flat = []
                    for frame_kp in seq_buffer:
                        seq_flat += frame_kp
                    writer.writerow([series_uuid, label] + seq_flat)
                    seq_buffer.clear()
                    print(f"✅ Lưu series_id={series_uuid}, label={label}")
                    
                    # Reset counters
                    pose_detected = 0
                    left_hand_detected = 0
                    right_hand_detected = 0
                    break
                else:
                    print("⚠️ Phím không hợp lệ, bỏ qua series này.")
                    seq_buffer.clear()
                    pose_detected = 0
                    left_hand_detected = 0
                    right_hand_detected = 0
                    break

        frame_idx += 1

cap.release()
cv2.destroyAllWindows()
holistic.close()
print(f"\n📊 Total detection: Pose={pose_detected}, Left Hand={left_hand_detected}, Right Hand={right_hand_detected}")