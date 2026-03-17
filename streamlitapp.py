import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import joblib
import tensorflow as tf
from collections import deque
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import base64

# --- Streamlit Page Config ---
st.set_page_config(page_title="LSTM Drowsiness Detector", layout="wide")
st.title("Real-Time Drowsiness Detection System")

# --- Constants ---
MODEL_PATH = "drowsiness_model.h5"
SCALER_PATH = "scaler.pkl"
SEQ_LEN = 30                
ALERT_CONSEC_FRAMES = 10     
PRED_THRESHOLD = 0.7         
PITCH_DROWSY_THRESHOLD = -160.0 

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_INNER = [78, 308, 303, 73, 12, 11]
MOUTH_OUTER = [61, 291, 0, 17]
POSE_LANDMARKS = [1, 152, 133, 362, 61, 291]
face_model_3d = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
], dtype=np.float64)

# --- Load Models Globally (Cached) ---
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return model, scaler, mp_face_mesh

model, scaler, face_mesh = load_models()

# --- Helper Functions ---
def dist(p1, p2): return math.hypot(p1.x - p2.x, p1.y - p2.y)

def ear(pts):
    try: return (dist(pts[1], pts[5]) + dist(pts[2], pts[4])) / (2 * dist(pts[0], pts[3]))
    except ZeroDivisionError: return 0.0

def mar(pts):
    try: return dist(pts[2], pts[3]) / dist(pts[0], pts[1])
    except ZeroDivisionError: return 0.0

def mouth_distance(pts):
    try: return dist(pts[0], pts[1])
    except ZeroDivisionError: return 0.0

def euler(rot_vec):
    R, _ = cv2.Rodrigues(rot_vec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x, y, z = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0
    else:
        x, y, z = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    return math.degrees(x), math.degrees(y), math.degrees(z)

class BlinkDetector:
    def __init__(self, ear_thresh=0.30, consec_frames=3):
        self.ear_thresh = ear_thresh
        self.consec_frames = consec_frames
        self.counter = 0
        self.blink_start_time = 0

    def update(self, ear_val):
        duration_to_return = 0.0
        if ear_val < self.ear_thresh:
            if self.counter == 0: self.blink_start_time = time.time()
            self.counter += 1
        else:
            if self.counter >= self.consec_frames:
                duration_to_return = time.time() - self.blink_start_time
            self.counter = 0
            self.blink_start_time = 0
        return duration_to_return

# --- WebRTC Video Processor ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.data_buffer = deque(maxlen=SEQ_LEN)
        self.consec_alert_frames = 0
        self.blink_detector = BlinkDetector(ear_thresh=0.30, consec_frames=3)
        self.status = "STARTING..."
        self.color = (0, 255, 255)
        self.is_drowsy = False 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

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
            blink_duration = self.blink_detector.update(avg_ear)

            pts2d = np.array([[int(p.x * w), int(p.y * h)] for p in pose_pts], dtype=np.float64)
            cam_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
            
            try:
                _, rot, _ = cv2.solvePnP(face_model_3d, pts2d, cam_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
                pitch, _, _ = euler(rot)
            except Exception:
                pitch = 0.0

            current_features = [avg_ear, mar_val, pitch, blink_duration]
            scaled_features = scaler.transform([current_features])[0]
            self.data_buffer.append(scaled_features)

            if len(self.data_buffer) == SEQ_LEN:
                X = np.expand_dims(self.data_buffer, axis=0)
                pred = model.predict(X, verbose=0)[0]
                drowsy_prob = pred[1] 
                
                pitch_is_drowsy = (pitch > PITCH_DROWSY_THRESHOLD)
                
                if (drowsy_prob > PRED_THRESHOLD) or pitch_is_drowsy:
                    self.consec_alert_frames += 1
                else:
                    self.consec_alert_frames = max(0, self.consec_alert_frames - 1)

                if self.consec_alert_frames >= ALERT_CONSEC_FRAMES:
                    self.status = "DROWSY - WAKE UP!"
                    self.color = (0, 0, 255) 
                    self.is_drowsy = True 
                else:
                    self.status = "ALERT"
                    self.color = (0, 255, 0) 
                    self.is_drowsy = False 

                conf_text = f"LSTM Conf: {drowsy_prob:.2f}"
                cv2.putText(img, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            else:
                self.status = f"WARMING UP LSTM... {len(self.data_buffer)}/{SEQ_LEN}"
                self.color = (0, 255, 255)
                self.is_drowsy = False

            # Draw Debug Info
            debug_color = (255, 255, 0) 
            cv2.putText(img, f"EAR: {avg_ear:.2f} | MAR: {mar_val:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
            cv2.putText(img, f"Pitch: {pitch:.1f} | Blink: {blink_duration:.2f}s", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)

        else:
            self.data_buffer.clear()
            self.consec_alert_frames = 0
            self.blink_detector.counter = 0
            self.blink_detector.blink_start_time = 0
            self.status = "NO FACE DETECTED"
            self.color = (0, 0, 255)
            self.is_drowsy = False

        cv2.putText(img, f"Status: {self.status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Audio Alarm Function ---
def get_audio_html(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f'<audio autoplay="true" src="data:audio/wav;base64,{b64}"></audio>'
    except FileNotFoundError:
        return "" # Fail silently if the audio file isn't found

# --- Main Streamlit Execution ---
st.write("Click 'Start' to activate your webcam and begin drowsiness detection.")

ctx = webrtc_streamer(
    key="drowsiness",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

audio_placeholder = st.empty()

# Background loop to check for drowsiness and play sound
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            if ctx.video_processor.is_drowsy:
                audio_placeholder.markdown(get_audio_html("alarm.wav"), unsafe_allow_html=True)
                time.sleep(1.5) # Throttle the alarm playback
            else:
                audio_placeholder.empty()
        time.sleep(0.5)