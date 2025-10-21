import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from ultralytics import YOLO
import uuid
import tensorflow as tf
import joblib

# === Cấu hình nhãn ===
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
    9: "STRIKING_WITH_HAND_OR_KNEE",
    10: "USE_COMPUTER"
}

# === Đường dẫn ===
video_path = "3761698845-preview.mp4"
output_csv = "dataset/keypoints_image_series.csv"
model_path = "pose_classifier_dense_series.tflite"
encoder_path = "label_encoder_dense_series.pkl"

# === Load TFLite Model ===
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    label_encoder = joblib.load(encoder_path)
    model_loaded = True
    print("✅ Model đã load thành công!")
except Exception as e:
    print(f"⚠️ Không load được model: {e}")
    model_loaded = False

# === MediaPipe config ===
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                             min_detection_confidence=0.5, min_tracking_confidence=0.3)
hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                               min_detection_confidence=0.5, min_tracking_confidence=0.3)

# === YOLO ===
yolo_model = YOLO("yolo11l.pt")

# === Cấu trúc keypoints ===
num_pose = 33
num_hand = 21
num_objects = 5  # số lượng object tối đa lưu
total_kp = num_pose + 2 * num_hand + num_objects  # tổng số keypoints x/y (pose+hands+objects)

# === CSV header ===
header = ["series_id", "label"] \
         + [f"x{i}" for i in range(total_kp)] \
         + [f"y{i}" for i in range(total_kp)] \
         + [f"obj_cls{i}" for i in range(num_objects)] \
         + [f"hand_obj_overlap{i}" for i in range(num_objects)]
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
if not os.path.exists(output_csv):
    with open(output_csv, "w", newline="") as f:
        csv.writer(f).writerow(header)

# === Hàm chuẩn hóa landmarks ===
def normalize_landmarks(landmarks, num_expected):
    if not landmarks:
        return [0] * (num_expected * 2)
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    norm = []
    for lm in landmarks:
        norm += [(lm.x - min_x)/width, (lm.y - min_y)/height]
    return norm

# === Hàm predict ===
def predict_tflite(pose_lms, left_lms, right_lms, obj_points, obj_cls, obj_bboxes, frame_w, frame_h):
    """
    pose_lms: pose landmarks (MediaPipe)
    left_lms, right_lms: hand landmarks
    obj_points: list center object [(cx,cy)]
    obj_cls: list class id của object
    obj_bboxes: list bounding box của object [(x1,y1,x2,y2)]
    frame_w, frame_h: kích thước frame
    """
    # --- Normalize landmarks ---
    kp_vector = []
    kp_vector += normalize_landmarks(pose_lms, num_pose)
    kp_vector += normalize_landmarks(left_lms, num_hand)
    kp_vector += normalize_landmarks(right_lms, num_hand)

    # --- Object centers ---
    save_obj_centers = []
    save_cls_ids = []
    save_interactions = []
    for (cx, cy), cid, bbox in zip(obj_points, obj_cls, obj_bboxes):
        save_obj_centers += [cx, cy]
        save_cls_ids.append(cid)
        # Hand-object overlap
        overlap = max(
            hand_object_overlap(left_lms, bbox, frame_w, frame_h),
            hand_object_overlap(right_lms, bbox, frame_w, frame_h)
        )
        save_interactions.append(overlap)

    # --- Padding nếu thiếu ---
    while len(save_cls_ids) < num_objects:
        save_obj_centers += [0,0]
        save_cls_ids.append(0)
        save_interactions.append(0)

    # --- Full vector 170 chiều ---
    kp_vector += save_obj_centers + save_cls_ids + save_interactions

    # --- Convert & predict ---
    kp_array = np.array(kp_vector, dtype=np.float32).reshape(1, -1)
    if kp_array.shape[1] != 170:
        return "INVALID_SHAPE", 0.0
    interpreter.set_tensor(input_details[0]['index'], kp_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(output)
    return label_encoder.classes_[idx], output[idx]

# === Hàm check hand-object overlap ===
def hand_object_overlap(hand_lms, obj_bbox, frame_w, frame_h):
    if hand_lms is None or not obj_bbox:
        return 0
    x1, y1, x2, y2 = obj_bbox
    for lm in hand_lms:
        hx, hy = int(lm.x*frame_w), int(lm.y*frame_h)
        if x1 <= hx <= x2 and y1 <= hy <= y2:
            return 1
    return 0

# === Mở video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"⚠️ Không mở được video: {video_path}")
    exit()

series_id = str(uuid.uuid4())
frame_index = 0

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]

        # === Pose ===
        pose_res = pose_detector.process(frame_rgb)

        # === Hand ===
        hand_res = hand_detector.process(frame_rgb)
        left_landmarks, right_landmarks = None, None
        if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            for idx_h, hand_label in enumerate(hand_res.multi_handedness):
                label = hand_label.classification[0].label
                hand_lms = hand_res.multi_hand_landmarks[idx_h]
                if label == "Left":
                    left_landmarks = hand_lms
                else:
                    right_landmarks = hand_lms

        # === YOLO Object Detection ===
        yolo_res = yolo_model.predict(frame_rgb, verbose=False)
        obj_points = []
        obj_cls = []
        obj_bboxes = []
        if len(yolo_res) > 0:
            for box in yolo_res[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = yolo_model.names[cls_id]
                if cls_name.lower() == "person":
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_points.append(((x1+x2)/2/frame_w, (y1+y2)/2/frame_h))
                obj_cls.append(cls_id)
                obj_bboxes.append((x1, y1, x2, y2))
                # Draw object
                color = (255,0,255) if cls_name.lower() in {"tv","laptop","keyboard","mouse"} else (0,255,0)
                cv2.rectangle(frame, (x1,y1),(x2,y2), color,2)
                cv2.putText(frame, f"{cls_name} {cls_id}", (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                if len(obj_points) >= num_objects:
                    break

        # --- Padding objects ---
        while len(obj_points) < num_objects:
            obj_points.append((0,0))
            obj_cls.append(0)
            obj_bboxes.append(None)

        # --- Build keypoint vector for prediction ---
        kp_vector = []
        kp_vector += normalize_landmarks(pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else [], num_pose)
        kp_vector += normalize_landmarks(left_landmarks.landmark if left_landmarks else [], num_hand)
        kp_vector += normalize_landmarks(right_landmarks.landmark if right_landmarks else [], num_hand)

        for (cx, cy), cid in zip(obj_points, obj_cls):
            kp_vector += [cx, cy]

        # Predict action
        predicted_label, confidence = predict_tflite(
    pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None,
    left_landmarks.landmark if left_landmarks else None,
    right_landmarks.landmark if right_landmarks else None,
    obj_points,
    obj_cls,
    obj_bboxes,
    frame_w,
    frame_h
)
        if model_loaded:
            pred_text = f"PREDICTION: {predicted_label} ({confidence*100:.1f}%)"
            pred_color = (0, 255, 255) if confidence > 0.7 else (0, 165, 255)
            cv2.putText(frame, pred_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)

        # --- Draw landmarks ---
        if pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if left_landmarks:
            mp_drawing.draw_landmarks(frame, left_landmarks, mp_hands.HAND_CONNECTIONS)
        if right_landmarks:
            mp_drawing.draw_landmarks(frame, right_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- Resize if too large ---
        max_height = 900
        if frame_h > max_height:
            scale = max_height / frame_h
            frame = cv2.resize(frame, (int(frame_w*scale), max_height))

        cv2.imshow("Pose + Hand + Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # --- Input label ---
        user_input = input("➡️ Nhập số label (0–10, ESC để thoát, ENTER dùng prediction): ").strip()
        if user_input.lower() == 'esc':
            break
        if user_input == "" and model_loaded:
            label = predicted_label
        elif user_input.isdigit() and int(user_input) in LABELS:
            label = LABELS[int(user_input)]
        else:
            print("⚠️ Bỏ qua frame này.")
            continue

        # --- Build CSV vector with hand-object overlap ---
        save_kp_vector = []
        save_kp_vector += normalize_landmarks(pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else [], num_pose)
        save_kp_vector += normalize_landmarks(left_landmarks.landmark if left_landmarks else [], num_hand)
        save_kp_vector += normalize_landmarks(right_landmarks.landmark if right_landmarks else [], num_hand)

        save_obj_centers = []
        save_cls_ids = []
        save_interactions = []

        for (cx, cy), cid, bbox in zip(obj_points, obj_cls, obj_bboxes):
            save_obj_centers += [cx, cy]
            save_cls_ids.append(cid)
            overlap = max(
                hand_object_overlap(left_landmarks.landmark if left_landmarks else None, bbox, frame_w, frame_h),
                hand_object_overlap(right_landmarks.landmark if right_landmarks else None, bbox, frame_w, frame_h)
            )
            save_interactions.append(overlap)

        # Padding if needed
        while len(save_cls_ids) < num_objects:
            save_obj_centers += [0,0]
            save_cls_ids.append(0)
            save_interactions.append(0)

        csv_vector = save_kp_vector + save_obj_centers + save_cls_ids + save_interactions
        writer.writerow([series_id, label] + csv_vector)
        print(f"✅ Lưu frame {frame_index} với label={label}")

cap.release()
cv2.destroyAllWindows()
pose_detector.close()
hand_detector.close()
print(f"\n✅ Hoàn tất trích xuất keypoints & nhập nhãn từ video.")
print(f"📁 Lưu tại: {output_csv}")
