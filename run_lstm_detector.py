import cv2
import mediapipe as mp
import numpy as np
import math
import joblib
import tensorflow as tf
from collections import deque
import time
import winsound

MODEL_PATH = "drowsiness_model.h5"
SCALER_PATH = "scaler.pkl"
SEQ_LEN = 30                
ALERT_CONSEC_FRAMES = 10     
PRED_THRESHOLD = 0.7         
SHOW_FPS = True

# Normal = -179, Forward = -150, Backward = 170
# Any pitch value > than -160 considered a drowsy tilt
PITCH_DROWSY_THRESHOLD = -160.0 

try:
    print("[INFO] Loading LSTM model...")
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit()

try:
    print("[INFO] Loading feature scaler...")
    scaler = joblib.load(SCALER_PATH)
    print(f"[DEBUG] Scaler expects {scaler.n_features_in_} features.")
except Exception as e:
    print(f"[ERROR] Could not load scaler: {e}")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_INNER = [78, 308, 303, 73, 12, 11]
MOUTH_OUTER = [61, 291, 0, 17]
POSE_LANDMARKS = [1, 152, 133, 362, 61, 291]
face_model_3d = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
], dtype=np.float64)

def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def ear(pts):
    try:
        return (dist(pts[1], pts[5]) + dist(pts[2], pts[4])) / (2 * dist(pts[0], pts[3]))
    except ZeroDivisionError:
        return 0.0

def mar(pts):
    try:
        return dist(pts[2], pts[3]) / dist(pts[0], pts[1])
    except ZeroDivisionError:
        return 0.0

def mouth_distance(pts):
    try:
        return dist(pts[0], pts[1])
    except ZeroDivisionError:
        return 0.0

def euler(rot_vec):
    R, _ = cv2.Rodrigues(rot_vec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    else:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(x), math.degrees(y), math.degrees(z)

class BlinkDetector:
    def __init__(self, ear_thresh=0.22, consec_frames=3):
        self.ear_thresh = ear_thresh
        self.consec_frames = consec_frames
        self.counter = 0
        self.blink_start_time = 0

    def update(self, ear_val):
        duration_to_return = 0.0
        if ear_val < self.ear_thresh:
            if self.counter == 0:
                self.blink_start_time = time.time()
            self.counter += 1
        else:
            if self.counter >= self.consec_frames:
                duration_to_return = time.time() - self.blink_start_time
            self.counter = 0
            self.blink_start_time = 0
        return duration_to_return

cap = cv2.VideoCapture(0)
time.sleep(1.0)

data_buffer = deque(maxlen=SEQ_LEN)
fps_hist = deque(maxlen=30)
consec_alert_frames = 0

# Using 0.30 (0.45 open, 0.20 closed)
blink_detector = BlinkDetector(ear_thresh=0.30, consec_frames=3)

print("[INFO] Running detector... press 'q' to quit")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    status = "NO FACE"
    color = (0, 0, 255) 
    pitch = 0.0 
    
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        
        leye_pts = [lm[i] for i in LEFT_EYE]
        reye_pts = [lm[i] for i in RIGHT_EYE]
        mouth_pts = [lm[i] for i in MOUTH_INNER]
        mouth_outer_pts = [lm[i] for i in MOUTH_OUTER]
        pose_pts = [lm[i] for i in POSE_LANDMARKS]
        
        avg_ear = (ear(leye_pts) + ear(reye_pts)) / 2.0
        mar_val = mar(mouth_outer_pts)
        mouth_dist = mouth_distance(mouth_pts)
        blink_duration = blink_detector.update(avg_ear)

        pts2d = np.array([[int(p.x * w), int(p.y * h)] for p in pose_pts], dtype=np.float64)
        cam_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        
        try:
            _, rot, _ = cv2.solvePnP(face_model_3d, pts2d, cam_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
            pitch, _, _ = euler(rot)
        except Exception:
            pitch = 0.0

        current_features = [avg_ear, mar_val, pitch, blink_duration]
        
        scaled_features = scaler.transform([current_features])[0]
        data_buffer.append(scaled_features)

        if len(data_buffer) == SEQ_LEN:
            X = np.expand_dims(data_buffer, axis=0)
            
            pred = model.predict(X, verbose=0)[0]
            
            drowsy_prob = pred[1] 
            
            # Test: Normal=-179 (False), Forward=-150 (True), Backward=170 (True)
            pitch_is_drowsy = (pitch > PITCH_DROWSY_THRESHOLD)
            
            if (drowsy_prob > PRED_THRESHOLD) or pitch_is_drowsy:
                consec_alert_frames += 1
            else:
                consec_alert_frames = max(0, consec_alert_frames - 1)

            if consec_alert_frames >= ALERT_CONSEC_FRAMES:
                status = "DROWSY"
                color = (0, 0, 255) # Red
                if consec_alert_frames == ALERT_CONSEC_FRAMES:
                    try: winsound.Beep(1000, 300)
                    except: pass
            else:
                status = "ALERT"
                color = (0, 255, 0) # Green

            conf_text = f"Conf: {drowsy_prob:.2f}"
            cv2.putText(frame, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        else:
            status = f"WARMING UP... {len(data_buffer)}/{SEQ_LEN}"
            color = (0, 255, 255) # Yellow
            consec_alert_frames = 0

        debug_color = (255, 255, 0) # Cyan
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
        cv2.putText(frame, f"MAR: {mar_val:.2f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
        cv2.putText(frame, f"BlinkDur: {blink_duration:.2f}", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)

    else:
        # No face 
        data_buffer.clear()
        consec_alert_frames = 0
        blink_detector.counter = 0
        blink_detector.blink_start_time = 0

    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # FPS
    fps = 1 / (time.time() - t0)
    fps_hist.append(fps)
    if SHOW_FPS:
        cv2.putText(frame, f"FPS: {np.mean(fps_hist):.1f}", (w - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("LSTM Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()