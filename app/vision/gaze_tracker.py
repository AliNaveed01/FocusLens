import cv2
import mediapipe as mp
import numpy as np
import time

class GazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = [469, 470, 471, 472]
        self.RIGHT_IRIS = [474, 475, 476, 477]

        self.eye_open_threshold = 0.15
        self.frames_threshold = 10
        self.closed_eye_count = 0
        self.gaze_history = []

        self.session_active = False
        self.session_start_time = None

    def calculate_ear(self, eye):
        vertical_1 = np.linalg.norm(eye[1] - eye[5])
        vertical_2 = np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def calculate_iris_position(self, eye_landmarks, iris_landmarks):
        eye_width = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        iris_center = np.mean(iris_landmarks, axis=0)
        iris_to_left_corner = np.linalg.norm(iris_center - eye_landmarks[0])
        return iris_to_left_corner / eye_width

    def calculate_orientation(self, landmarks):
        nose = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        chin = np.array([landmarks[152].x, landmarks[152].y, landmarks[152].z])
        left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
        right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])

        eye_vector = right_eye - left_eye
        face_vector = chin - nose

        yaw = np.arctan2(eye_vector[2], eye_vector[0]) * 180 / np.pi
        pitch = np.arctan2(face_vector[2], face_vector[1]) * 180 / np.pi
        roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi

        return yaw, pitch, roll

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                left_eye = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in self.LEFT_EYE])
                right_eye = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in self.RIGHT_EYE])

                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)

                if left_ear < self.eye_open_threshold and right_ear < self.eye_open_threshold:
                    self.closed_eye_count += 1
                    if self.closed_eye_count >= self.frames_threshold:
                        cv2.putText(frame, "Status: Eyes Closed", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        return frame
                else:
                    self.closed_eye_count = 0
                    cv2.putText(frame, "Status: Eyes Open", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                left_iris = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in self.LEFT_IRIS])
                right_iris = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in self.RIGHT_IRIS])

                yaw, pitch, roll = self.calculate_orientation(landmarks.landmark)

                # Adjust gaze thresholds dynamically
                left_pos = self.calculate_iris_position(left_eye, left_iris)
                right_pos = self.calculate_iris_position(right_eye, right_iris)

                lmin, lmax = 0.35, 0.65
                rmin, rmax = 0.35, 0.60
                min_off, max_off = 0, 0

                if yaw < -25:
                    min_off, max_off = 0.05, 0.10
                elif yaw > 20:
                    min_off, max_off = -0.05, -0.10

                if (lmin + min_off) < left_pos < (lmax + max_off) and (rmin + min_off) < right_pos < (rmax + max_off):
                    gaze = "Looking at Camera"
                else:
                    gaze = "Looking Away"

                self.gaze_history.append(gaze)
                if len(self.gaze_history) > 10:
                    self.gaze_history.pop(0)

                if self.gaze_history.count("Looking Away") >= 4:
                    final_gaze = "Looking Away"
                    if self.session_active:
                        self.session_active = False
                        end_time = time.time()
                        print(f"Session ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
                        print(f"Session duration: {end_time - self.session_start_time:.2f} seconds")
                else:
                    final_gaze = "Looking at Camera"
                    if not self.session_active:
                        self.session_active = True
                        self.session_start_time = time.time()
                        print(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.session_start_time))}")

                cv2.putText(frame, f"Gaze Status: {final_gaze}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Visualization
                for point in np.concatenate((left_eye, right_eye), axis=0):
                    cv2.circle(frame, tuple(int(v) for v in point), 2, (255, 0, 0), -1)
                for point in np.concatenate((left_iris, right_iris), axis=0):
                    cv2.circle(frame, tuple(int(v) for v in point), 2, (0, 255, 0), -1)

        return frame


def run_gaze_tracker():
    cap = cv2.VideoCapture(0)
    tracker = GazeTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = tracker.process_frame(frame)
        cv2.imshow("Gaze Tracking", processed)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
