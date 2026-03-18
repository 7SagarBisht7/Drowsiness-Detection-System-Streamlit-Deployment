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
st.set_page_config(page_title="LSTM Drowsiness Detector", layout="centered")
st.title("Real-Time Drowsiness Detection System")

# --- Constants (Tuned for WebRTC Latency) ---
MODEL_PATH = "drowsiness_model.h5"
SCALER_PATH = "scaler.pkl"
SEQ_LEN = 30                
ALERT_CONSEC_FRAMES = 5      
PRED_THRESHOLD = 0.55        
PITCH_DROWSY_THRESHOLD = -160.0 

# NEW: Tuned Thresholds for overrides
EAR_THRESH = 0.25             # Hard threshold for eyes closed
MICRO_SLEEP_TIME = 1.0        # Seconds of eyes closed to instantly trigger alarm
HEAD_DROP_FRAMES = 5          # Frames without face to trigger head-drop alarm

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
        refine_landmarks=False, 
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

def euler(rot_vec):
    R, _ = cv2.Rodrigues(rot_vec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x, y, z = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0
    else:
        x, y, z = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    return math.degrees(x), math.degrees(y), math.degrees(z)

class BlinkDetector:
    def __init__(self, ear_thresh=EAR_THRESH):
        self.ear_thresh = ear_thresh
        self.counter = 0
        self.blink_start_time = 0

    def update(self, ear_val):
        # FIX 1: Live stopwatch. Now it reports time continually WHILE eyes are closed.
        if ear_val < self.ear_thresh:
            if self.counter == 0: 
                self.blink_start_time = time.time()
            self.counter += 1
            return time.time() - self.blink_start_time
        else:
            self.counter = 0
            self.blink_start_time = 0
            return 0.0

# --- WebRTC Video Processor ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.data_buffer = deque(maxlen=SEQ_LEN)
        self.consec_alert_frames = 0
        self.missing_face_frames = 0  # NEW: Tracks head drops
        self.blink_detector = BlinkDetector(ear_thresh=EAR_THRESH)
        
        self.status = "STARTING..."
        self.color = (0, 255, 255)
        self.is_drowsy = False 
        
        self.frame_count = 0
        self.last_drowsy_prob = 0.0
        self.last_avg_ear = 0.0
        self.last_mar_val = 0.0
        self.last_pitch = 0.0
        self.last_blink = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 2 != 0:
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                self.missing_face_frames = 0 # Reset missing counter
                lm = res.multi_face_landmarks[0].landmark
                
                leye_pts = [lm[i] for i in LEFT_EYE]
                reye_pts = [lm[i] for i in RIGHT_EYE]
                mouth_pts = [lm[i] for i in MOUTH_INNER]
                mouth_outer_pts = [lm[i] for i in MOUTH_OUTER]
                pose_pts = [lm[i] for i in POSE_LANDMARKS]
                
                self.last_avg_ear = (ear(leye_pts) + ear(reye_pts)) / 2.0
                self.last_mar_val = mar(mouth_outer_pts)
                self.last_blink = self.blink_detector.update(self.last_avg_ear)

                pts2d = np.array([[int(p.x * w), int(p.y * h)] for p in pose_pts], dtype=np.float64)
                cam_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
                
                try:
                    _, rot, _ = cv2.solvePnP(face_model_3d, pts2d, cam_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
                    self.last_pitch, _, _ = euler(rot)
                except Exception:
                    self.last_pitch = 0.0

                current_features = [self.last_avg_ear, self.last_mar_val, self.last_pitch, self.last_blink]
                scaled_features = scaler.transform([current_features])[0]
                self.data_buffer.append(scaled_features)

                if len(self.data_buffer) == SEQ_LEN:
                    X = np.expand_dims(self.data_buffer, axis=0)
                    pred = model(X, training=False)[0].numpy()
                    self.last_drowsy_prob = pred[1] 
                    
                    # Hard Overrides
                    pitch_is_drowsy = (self.last_pitch > PITCH_DROWSY_THRESHOLD)
                    eyes_closed_drowsy = (self.last_blink >= MICRO_SLEEP_TIME)
                    
                    if (self.last_drowsy_prob > PRED_THRESHOLD) or pitch_is_drowsy or eyes_closed_drowsy:
                        self.consec_alert_frames += 1
                    else:
                        self.consec_alert_frames = 0 # Instant reset to normal when awake

                    if self.consec_alert_frames >= ALERT_CONSEC_FRAMES:
                        self.status = "DROWSY - WAKE UP!"
                        self.color = (0, 0, 255) 
                        self.is_drowsy = True 
                    else:
                        self.status = "ALERT"
                        self.color = (0, 255, 0) 
                        self.is_drowsy = False 
            else:
                # FIX 2: HEAD DROP LOGIC
                self.missing_face_frames += 1
                self.blink_detector.counter = 0
                self.blink_detector.blink_start_time = 0
                
                if self.missing_face_frames >= HEAD_DROP_FRAMES:
                    self.status = "DROWSY - HEAD DROP!"
                    self.color = (0, 0, 255)
                    self.is_drowsy = True
                else:
                    self.status = "SEARCHING FOR FACE..."
                    self.color = (0, 165, 255) # Orange warning
                    self.is_drowsy = False

        # --- DRAWING LAYER ---
        cv2.putText(img, f"Status: {self.status}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)
        
        if len(self.data_buffer) == SEQ_LEN and self.missing_face_frames == 0:
            cv2.putText(img, f"LSTM Conf: {self.last_drowsy_prob:.2f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.missing_face_frames == 0:
            cv2.putText(img, f"WARMING UP... {len(self.data_buffer)}/{SEQ_LEN}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.missing_face_frames == 0:
            debug_color = (255, 255, 0) 
            cv2.putText(img, f"EAR: {self.last_avg_ear:.2f} | MAR: {self.last_mar_val:.2f}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
            cv2.putText(img, f"Pitch: {self.last_pitch:.1f} | Blink: {self.last_blink:.2f}s", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Audio Alarm Function ---
def get_audio_html(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            # loop="true" ensures it keeps beeping until explicitly cleared
            return f'<audio autoplay="true" loop="true" src="data:audio/wav;base64,{b64}"></audio>'
    except FileNotFoundError:
        return "" 

# --- Main Streamlit Execution ---
st.write("Click 'Start' to activate your webcam and begin drowsiness detection.")

ctx = webrtc_streamer(
    key="drowsiness",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True, 
        "audio": False
    },
    async_processing=True 
)

audio_placeholder = st.empty()
alarm_playing = False

# Background Audio Loop
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            if ctx.video_processor.is_drowsy and not alarm_playing:
                # Turn alarm ON
                audio_placeholder.markdown(get_audio_html("alert.wav"), unsafe_allow_html=True)
                alarm_playing = True
            
            elif not ctx.video_processor.is_drowsy and alarm_playing:
                # Turn alarm OFF instantly
                audio_placeholder.empty()
                alarm_playing = False
                
        time.sleep(0.2) # Fast poll to keep visual and audio perfectly synced
