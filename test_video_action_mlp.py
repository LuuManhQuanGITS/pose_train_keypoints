import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from ultralytics import YOLO

# --- Config ---
video_path = "1058341822-preview.mp4"
confidence_threshold = 0.8
max_height_display = 900
num_objects = 5
num_pose = 33
num_hand = 21

# --- Load TFLite model + label encoder ---
interpreter = tf.lite.Interpreter(model_path="pose_classifier_dense_series.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
label_encoder = joblib.load("label_encoder_dense_series.pkl")

# --- MediaPipe ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                             min_detection_confidence=0.5, min_tracking_confidence=0.3)
hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                               min_detection_confidence=0.5, min_tracking_confidence=0.3)

# --- YOLO ---
yolo_model = YOLO("yolo11l.pt")

# --- Normalize landmarks ---
def normalize_landmarks(landmarks, num_expected):
    if not landmarks:
        return [0]*(num_expected*2)
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(max_x - min_x, 1e-6)
    h = max(max_y - min_y, 1e-6)
    norm = []
    for lm in landmarks:
        norm += [(lm.x - min_x)/w, (lm.y - min_y)/h]
    return norm

# --- Check hand-object overlap ---
def hand_object_overlap(hand_lms, obj_bbox, frame_w, frame_h):
    if hand_lms is None or obj_bbox is None:
        return 0
    x1, y1, x2, y2 = obj_bbox
    for lm in hand_lms:
        hx, hy = int(lm.x*frame_w), int(lm.y*frame_h)
        if x1 <= hx <= x2 and y1 <= hy <= y2:
            return 1
    return 0

# --- Predict with TFLite ---
def predict_tflite(kp_vector):
    kp_array = np.array(kp_vector, dtype=np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], kp_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(output)
    return label_encoder.classes_[idx], output[idx]

# --- Video Capture ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"⚠️ Cannot open video: {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_h, frame_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Pose + Hands ---
    pose_res = pose_detector.process(frame_rgb)
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

    # --- YOLO Object Detection ---
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

    # --- Build keypoint vector 170 dim ---
    kp_vector = []
    kp_vector += normalize_landmarks(pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else [], num_pose)
    kp_vector += normalize_landmarks(left_landmarks.landmark if left_landmarks else [], num_hand)
    kp_vector += normalize_landmarks(right_landmarks.landmark if right_landmarks else [], num_hand)

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

    kp_vector += save_obj_centers + save_cls_ids + save_interactions  # 170 dim

    # --- Predict action ---
    try:
        predicted_label, confidence = predict_tflite(kp_vector)
        if confidence > confidence_threshold:
            cv2.putText(frame, f"{predicted_label}: {confidence*100:.1f}%", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    except Exception as e:
        cv2.putText(frame, f"Prediction error", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        print(e)

    # --- Draw landmarks ---
    if pose_res.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if left_landmarks:
        mp_drawing.draw_landmarks(frame, left_landmarks, mp_hands.HAND_CONNECTIONS)
    if right_landmarks:
        mp_drawing.draw_landmarks(frame, right_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- Resize frame if too large ---
    if frame_h > max_height_display:
        scale = max_height_display / frame_h
        frame = cv2.resize(frame, (int(frame_w*scale), max_height_display))

    cv2.imshow("Action Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose_detector.close()
hand_detector.close()
print("✅ Done.")
