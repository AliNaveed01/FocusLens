import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict, deque
from time import time
from tracking.sort import Sort  # assuming sort.py is placed under app/tracking

class SpeechDetector:
    def __init__(self):
        # Initialize Mediapipe Face Mesh model
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Lip landmarks (for speech activity)
        self.INNER_LIPS_TOP = 13
        self.INNER_LIPS_BOTTOM = 14

        # Speech detection parameters
        self.FRAME_HISTORY = 10
        self.SPEECH_THRESHOLD = 2.0
        self.RATE_THRESHOLD = 0.5
        self.CONSEC_FRAMES = 10
        self.SILENCE_TIMEOUT = 2  # seconds
        self.EMA_ALPHA = 0.3

        # Per-person lip movement history
        self.lip_distances_history = defaultdict(lambda: deque(maxlen=self.FRAME_HISTORY))
        self.smoothed_distances = defaultdict(lambda: None)
        self.speech_events = defaultdict(lambda: {
            "speaking": False,
            "start_time": None,
            "last_speaking_time": None,
            "speech_frames": 0
        })

        # Tracker for face ID consistency
        self.tracker = Sort(max_age=30, min_hits=3)

    def calculate_lip_distance(self, top, bottom):
        return np.linalg.norm(top - bottom)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        detections = []

        # Step 1: Gather face bounding boxes for SORT
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                x_min, y_min, x_max, y_max = w, h, 0, 0
                for lm in face.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                detections.append([x_min, y_min, x_max, y_max, 1.0])  # score = 1.0

        detections = np.array(detections)
        tracked_faces = self.tracker.update(detections)

        # Step 2: Process each tracked face
        for x1, y1, x2, y2, face_id in tracked_faces:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            for face in results.multi_face_landmarks:
                top_lip = np.array([face.landmark[self.INNER_LIPS_TOP].x * w, face.landmark[self.INNER_LIPS_TOP].y * h])
                bottom_lip = np.array([face.landmark[self.INNER_LIPS_BOTTOM].x * w, face.landmark[self.INNER_LIPS_BOTTOM].y * h])
                
                # Lip movement calculation
                distance = self.calculate_lip_distance(top_lip, bottom_lip)
                prev = self.smoothed_distances[face_id]
                smoothed = distance if prev is None else self.EMA_ALPHA * distance + (1 - self.EMA_ALPHA) * prev
                self.smoothed_distances[face_id] = smoothed
                self.lip_distances_history[face_id].append(smoothed)

                # Analyze recent lip movement
                avg = np.mean(self.lip_distances_history[face_id])
                variation = np.std(self.lip_distances_history[face_id])
                speaking = self.speech_events[face_id]

                # Step 3: Detect active speech
                if avg > self.SPEECH_THRESHOLD and variation > self.RATE_THRESHOLD:
                    speaking["speech_frames"] += 1
                    if speaking["speech_frames"] >= self.CONSEC_FRAMES:
                        if not speaking["speaking"]:
                            speaking["speaking"] = True
                            speaking["start_time"] = time()
                            print(f"Person {face_id} started speaking at {speaking['start_time']:.2f}s")
                        speaking["last_speaking_time"] = time()
                    status = "Speaking"
                else:
                    speaking["speech_frames"] = 0
                    if speaking["speaking"]:
                        if time() - speaking["last_speaking_time"] > self.SILENCE_TIMEOUT:
                            speaking["speaking"] = False
                            end_time = time()
                            duration = end_time - speaking["start_time"]
                            print(f"Person {face_id} stopped speaking at {end_time:.2f}s, duration: {duration:.2f}s")
                    status = "Silent"

                # Step 4: Annotate frame
                cv2.circle(frame, tuple(int(v) for v in top_lip), 3, (0, 255, 0), -1)
                cv2.circle(frame, tuple(int(v) for v in bottom_lip), 3, (0, 0, 255), -1)
                cv2.putText(frame, f"Speech Status: {status}", (cx, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame


def run_speech_detection():
    cap = cv2.VideoCapture(0)
    detector = SpeechDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = detector.process_frame(frame)
        cv2.imshow("Speech Detection", processed)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
