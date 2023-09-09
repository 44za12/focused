import cv2
import dlib
import time
from imutils import face_utils
from utils import confidence_score, notify


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

FACE_ABSENT_THRESHOLD = 90
THRESHOLD = 0.6

def save_scores_to_file(scores_list, timestamps):
    with open("concentration_report.txt", "a") as f:
        for timestamp, score in zip(timestamps, scores_list):
            f.write(f"{timestamp}\t{score}\n")

def main():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    interval = 60
    data_samples = []
    scores_list = []
    timestamps = []
    current_score = 0.0
    face_not_detected_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) == 0:
                face_not_detected_counter += 1
            else:
                face_not_detected_counter = 0

            if face_not_detected_counter > FACE_ABSENT_THRESHOLD:
                current_score = 0.05

            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                score = confidence_score(shape)
                data_samples.append(score)
                scores_list.append(score)
                timestamps.append(time.time())

            if time.time() - start_time > interval:
                try:
                    current_score = sum(data_samples) / len(data_samples)
                except ZeroDivisionError:
                    current_score = 0
                data_samples = []
                save_scores_to_file(scores_list, timestamps)
                scores_list = []
                timestamps = []
                if current_score < THRESHOLD:
                    notify(
                        title="Concentration Alert!",
                        text=f"Your concentration score in the last minute is low: {current_score:.2f}"
                    )
                print("Notification displayed.")
                start_time = time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()