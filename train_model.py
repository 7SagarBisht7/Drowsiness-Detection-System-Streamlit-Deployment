import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import winsound


def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_ear(eye_landmarks):
    try:
        p1 = eye_landmarks[0]
        p2 = eye_landmarks[1]
        p3 = eye_landmarks[2]
        p4 = eye_landmarks[3]
        p5 = eye_landmarks[4]
        p6 = eye_landmarks[5]
        ver_dist1 = calculate_distance(p2, p6)
        ver_dist2 = calculate_distance(p3, p5)
        hor_dist = calculate_distance(p1, p4)
        ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
        return ear
    except Exception as e:
        return 0.0

def calculate_mar(mouth_landmarks):
    try:
        p1 = mouth_landmarks[0] # Left
        p2 = mouth_landmarks[1] # Right
        p3 = mouth_landmarks[2] # Top
        p4 = mouth_landmarks[3] # Bottom
        ver_dist = calculate_distance(p3, p4)
        hor_dist = calculate_distance(p1, p2)
        mar = ver_dist / hor_dist
        return mar
    except Exception as e:
        return 0.0

def get_euler_angles(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
    return math.degrees(x), math.degrees(y), math.degrees(z)

EAR_OPEN = 0.35
EAR_CLOSED = 0.15

MAR_NORMAL = 1.0 
MAR_YAWN = 1.5

PITCH_NORMAL = -170.0
PITCH_NOD = -150.0 



def normalize(value, v_min, v_max, inverted=False):
    """Normalizes a value from its own range to a 0.0-1.0 score."""
    if inverted:
        value = (v_max - value)
        v_min, v_max = (0, v_max - v_min)
    
    score = (value - v_min) / (v_max - v_min)
    return max(0.0, min(1.0, score))

# Smoothing
SMOOTHING_BUFFER_SIZE = 10 
confidence_buffer = deque(maxlen=SMOOTHING_BUFFER_SIZE)


LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [61, 291, 0, 17] # [Left, Right, Top, Bottom]
POSE_INDICES = [1, 152, 133, 362, 61, 291]
face_model_3d = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
], dtype=np.float64)


print("Starting camera feed...")
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

ALARM_ON = False
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    frame_h, frame_w, _ = image.shape
    
    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    image.flags.writeable = False 
    image = cv2.flip(image, 1) 
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    image.flags.writeable = True 
    
    ear_score, mar_score, pitch_score = 0.0, 0.0, 0.0
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            all_landmarks = face_landmarks.landmark
            
            # --- 1. EAR Calculation (Eyes) ---
            left_eye_pts = [all_landmarks[i] for i in LEFT_EYE_INDICES]
            right_eye_pts = [all_landmarks[i] for i in RIGHT_EYE_INDICES]
            avg_ear = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0
            
            # --- 2. MAR Calculation (Mouth) ---
            mouth_pts = [all_landmarks[i] for i in MOUTH_INDICES]
            mar = calculate_mar(mouth_pts)

            # --- 3. Head Pose Calculation (Tilt) ---
            pose_pts_2d = np.array([[int(all_landmarks[i].x * frame_w), int(all_landmarks[i].y * frame_h)] for i in POSE_INDICES], dtype=np.float64)
            (success, rotation_vector, _) = cv2.solvePnP(face_model_3d, pose_pts_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            pitch, yaw, roll = get_euler_angles(rotation_vector)
            
            
            ear_score = normalize(avg_ear, EAR_CLOSED, EAR_OPEN, inverted=True)
            mar_score = normalize(mar, MAR_NORMAL, MAR_YAWN)
            
            pitch_score = normalize(pitch, PITCH_NOD, PITCH_NORMAL, inverted=True)

            raw_confidence = max(ear_score, pitch_score, mar_score)
            
            confidence_buffer.append(raw_confidence)
            
            smoothed_confidence = sum(confidence_buffer) / len(confidence_buffer)
            
            
            print(f"Drowsiness Confidence: {smoothed_confidence:.2f}")
            
            
            cv2.putText(image, f"EAR: {avg_ear:.2f} (Score: {ear_score:.2f})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"MAR: {mar:.2f} (Score: {mar_score:.2f})", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Pitch: {pitch:.2f} (Score: {pitch_score:.2f})", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(image, f"CONFIDENCE: {smoothed_confidence:.2f}", (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if smoothed_confidence > 0.9:
                if not ALARM_ON:
                    ALARM_ON = True
                    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                cv2.putText(image, "DROWSINESS DETECTED!", (10, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                ALARM_ON = False


    cv2.imshow('Drowsiness Detector - Confidence Score', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()