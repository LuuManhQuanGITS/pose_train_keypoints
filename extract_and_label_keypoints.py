import cv2
import mediapipe as mp
import csv
import os

# --- Danh sách nhãn ---
LABELS = {
    0: "UNKNOWN",
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

# --- Đường dẫn dữ liệu ---
image_folder = "dataset/images"
output_csv = "dataset/keypoints_labeled.csv"

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

# --- Khởi tạo Mediapipe ---
pose = mp_pose.Pose(static_image_mode=True)

# --- Tạo file CSV nếu chưa có ---
header = ["filename", "label"] + [f"x{i}" if i % 2 == 0 else f"y{i//2}" for i in range(66)]
if not os.path.exists(output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# --- Bắt đầu gán nhãn ---
with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)
    for filename in sorted(os.listdir(image_folder)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if not result.pose_landmarks:
            print(f"❌ Không phát hiện pose trong ảnh {filename}")
            continue

        # --- Vẽ keypoints ---
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- Hiển thị hướng dẫn ---
        cv2.imshow("Pose Labeling", image)
        print(f"\nẢnh: {filename}")
        for k, v in LABELS.items():
            print(f"{k}: {v}")
        print("Nhập số (1–9) để gán nhãn, hoặc nhấn '0' để bỏ qua, 'ESC' để thoát.")

        key = cv2.waitKey(0)

        if key == 27:  # ESC để thoát
            break
        elif chr(key).isdigit() and int(chr(key)) in LABELS:
            label = LABELS[int(chr(key))]
            print(f"✅ Gán nhãn: {label}")
        else:
            print("⚠️ Phím không hợp lệ, bỏ qua ảnh này.")
            continue

        # --- Chuẩn hóa và lưu keypoints ---
        keypoints = normalize_keypoints(result.pose_landmarks.landmark)
        writer.writerow([filename, label] + keypoints)

cv2.destroyAllWindows()
pose.close()
print("\n✅ Hoàn tất gán nhãn.")
