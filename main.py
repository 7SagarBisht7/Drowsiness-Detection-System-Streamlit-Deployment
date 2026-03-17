import cv2
import mediapipe as mp
import math
import numpy as np
import time

# Helper Functions (calculate_distance, calculate_ear, calculate_mar, get_euler_angles)
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_ear(eye_landmarks):
    try:
        p1 = eye_landmarks[0]; p2 = eye_landmarks[1]; p3 = eye_landmarks[2]
        p4 = eye_landmarks[3]; p5 = eye_landmarks[4]; p6 = eye_landmarks[5]
        ver_dist1 = calculate_distance(p2, p6)
        ver_dist2 = calculate_distance(p3, p5)
        hor_dist = calculate_distance(p1, p4)
        return (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
    except Exception as e: return 0.0

def calculate_mar(mouth_landmarks):
    try:
        p1 = mouth_landmarks[0]; p2 = mouth_landmarks[1]
        p3 = mouth_landmarks[2]; p4 = mouth_landmarks[3]
        ver_dist = calculate_distance(p3, p4)
        hor_dist = calculate_distance(p1, p2)
        return ver_dist / hor_dist
    except Exception as e: return 0.0

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


EAR_CLOSED_THRESHOLD = 0.2
is_blinking = False
blink_start_time = 0.0
last_blink_duration = 0.0

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [61, 291, 0, 17]
POSE_INDICES = [1, 152, 133, 362, 61, 291]
face_model_3d = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
], dtype=np.float64)


# MediaPipe and OpenCV Setup
print("Starting camera feed for DATA COLLECTION...")
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# store our feature vectors
feature_data_list = []
# -------------------------

# ---Main Loop---
while cap.isOpened():
    success, image = cap.read()
    if not success: continue

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
    
    avg_ear, mar, pitch = 0.3, 0.0, 0.0
    current_blink_duration = 0.0

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
            
            # --- 4. NEW: Blink Duration Logic ---
            if avg_ear < EAR_CLOSED_THRESHOLD:
                if not is_blinking:
                    # Blink just started
                    is_blinking = True
                    blink_start_time = time.time()
                # Store the *current* duration while eyes are closed
                current_blink_duration = time.time() - blink_start_time
            else:
                if is_blinking:
                    # Blink just ended
                    is_blinking = False
                    last_blink_duration = time.time() - blink_start_time
                # Reset current duration
                current_blink_duration = 0.0
            
            blink_feature = last_blink_duration

            feature_vector = [avg_ear, mar, pitch, blink_feature]
            feature_data_list.append(feature_vector)
            
            print(f"EAR: {avg_ear:.2f}, MAR: {mar:.2f}, Pitch: {pitch:.2f}, BlinkDur: {blink_feature:.2f}")

            cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Pitch: {pitch:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Blink (s): {blink_feature:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Drowsiness Data Collector', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

# Save data
import csv
print(f"Saving {len(feature_data_list)} frames of data to 'drowsiness_data.csv'...")

with open('drowsiness_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['EAR', 'MAR', 'Pitch', 'BlinkDuration'])
    writer.writerows(feature_data_list)

print("Data saved! Next steps: Label this data and train a model.")