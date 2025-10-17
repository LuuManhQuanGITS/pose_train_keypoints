import cv2
import mediapipe as mp
import csv
import os

# --- Danh sách nhãn ---
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

# --- Mediapipe Pose setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Video source ---
video_path = "1071362284-preview.mp4"  # đổi video của bạn
output_csv = "dataset/keypoints_video.csv"

# --- Hàm chuẩn hóa keypoints ---
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
    return norm_points

# --- Tạo CSV nếu chưa có ---
num_keypoints = 33  # MediaPipe Pose
header = ["label"] + [f"x{i}" if i % 2 == 0 else f"y{i//2}" for i in range(num_keypoints*2)]
if not os.path.exists(output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# --- Khởi tạo MediaPipe ---
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Mở video ---
cap = cv2.VideoCapture(video_path)
frame_idx = 0

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            # Vẽ pose
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Hiển thị frame để gán nhãn
            cv2.imshow("Label Pose Frame", frame)
            print(f"\nFrame: {frame_idx}")
            for k, v in LABELS.items():
                print(f"{k}: {v}")
            print("Nhập số (1–9) để gán nhãn, hoặc nhấn '0' để bỏ qua, 'ESC' để thoát.")

            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break

            elif chr(key).isdigit() and int(chr(key)) in LABELS:
                label = LABELS[int(chr(key))]
                print(f"✅ Gán nhãn: {label}")
            else:
                print("⚠️ Phím không hợp lệ, bỏ qua frame này.")
                frame_idx += 1
                continue

            # Chuẩn hóa keypoints và lưu
            keypoints = normalize_keypoints(result.pose_landmarks.landmark)
            writer.writerow([label] + keypoints)

        else:
            print(f"⚠️ Không phát hiện pose trong frame {frame_idx}")

        frame_idx += 1

cap.release()
cv2.destroyAllWindows()
pose.close()
print("\n✅ Hoàn tất extract keypoints từ video.")
