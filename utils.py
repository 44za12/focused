import numpy as np
import cv2
from imutils import face_utils
import subprocess

def notify(title, text):
    """Send an immediate alert on macOS."""
    try:
        script = f'display dialog "{text}" with title "{title}" buttons {{"OK"}} default button "OK"'
        subprocess.call(["osascript", "-e", script])
    except Exception as e:
        print(f"Notification error: {e}")

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[3] - mouth[9])  # 51 - 57 in dlib's 68-point model
    B = np.linalg.norm(mouth[2] - mouth[10]) # 50 - 58
    C = np.linalg.norm(mouth[4] - mouth[8])  # 52 - 56
    width = np.linalg.norm(mouth[0] - mouth[6])  # 48 - 54
    mar = (A + B + C) / (3.0 * width)
    return mar

def head_pose_estimation(landmarks):
    image_points = np.array([
        (landmarks[30][0], landmarks[30][1]),     # Nose tip
        (landmarks[8][0], landmarks[8][1]),       # Chin
        (landmarks[36][0], landmarks[36][1]),     # Left eye left corner
        (landmarks[45][0], landmarks[45][1]),     # Right eye right corner
        (landmarks[48][0], landmarks[48][1]),     # Left Mouth corner
        (landmarks[54][0], landmarks[54][1])      # Right mouth corner
    ], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-210.0, 170.0, -135.0),  # Left eye left corner
        (210.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])
    size = (640, 480)  # Default frame size
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return (rotation_vector, translation_vector)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def eyebrow_position(eyebrow, eye):
    return np.linalg.norm(eyebrow[2] - eye[1])

def mouth_openness(mouth):
    return np.linalg.norm(mouth[14] - mouth[18])

def confidence_score(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (lEBStart, lEBEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (rEBStart, rEBEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEyebrow = shape[lEBStart:lEBEnd]
    rightEyebrow = shape[rEBStart:rEBEnd]
    mouth = shape[mStart:mEnd]
    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
    ear = np.clip(ear, 0, 1)
    leb_score = 1 - (eyebrow_position(leftEyebrow, leftEye) / 100.0)
    leb_score = np.clip(leb_score, 0, 1)
    reb_score = 1 - (eyebrow_position(rightEyebrow, rightEye) / 100.0)
    reb_score = np.clip(reb_score, 0, 1)
    eyebrow_score = (leb_score + reb_score) / 2.0
    mouth_score = 1 - (mouth_openness(mouth) / 50.0)
    mouth_score = np.clip(mouth_score, 0, 1)
    rotation_vector, _ = head_pose_estimation(shape)
    rotation_magnitude = np.linalg.norm(rotation_vector)
    head_pose_score = 1 - (rotation_magnitude / np.pi)
    # head_pose_score = np.clip(head_pose_score, 0, 1)
    head_pose_score = 1
    combined_score = 0.25 * ear + 0.25 * eyebrow_score + 0.25 * mouth_score + 0.25 * head_pose_score
    return combined_score
