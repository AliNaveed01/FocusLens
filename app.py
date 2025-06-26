import cv2
import csv
import os
from datetime import datetime

import mediapipe as mp

from app.vision.gaze_tracker import GazeTracker
from app.vision.tilt_checker import HeadTiltChecker
from app.vision.speech_detector import SpeechDetector
from app.vision.distance_estimator import DistanceEstimator

# Prepare logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "session_log.csv")

# Initialize CSV and write header
with open(log_path, mode="w", newline="") as log_file:
    writer = csv.writer(log_file)
    writer.writerow([
        "timestamp",
        "gaze_direction",
        "eyes_closed",
        "tilt_direction",
        "speech_status",
        "distance_cm",
        "distance_category"
    ])

# Initialize modules
gaze_tracker = GazeTracker()
tilt_checker = HeadTiltChecker()
speech_detector = SpeechDetector()
distance_estimator = DistanceEstimator()

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam
cap = cv2.VideoCapture(0)
print("🚀 Vision Trackker started. Press ESC to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_frame)

    # Extract single face landmarks
    face_landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None

    # Initialize log values
    gaze_dir = "N/A"
    eyes_closed = "N/A"
    tilt_dir = "N/A"
    speech_status = "N/A"
    distance_cm = "N/A"
    distance_cat = "N/A"

    # --- Gaze Tracking ---
    if face_landmarks:
        processed_gaze, gaze_dir, eyes_closed = gaze_tracker.process_frame(frame.copy(), face_landmarks)
        frame = processed_gaze

    # --- Head Tilt ---
    processed_tilt = tilt_checker.process_frame(frame.copy())
    frame = processed_tilt
    # Note: tilt direction is drawn on frame already

    # --- Speech Detection ---
    processed_speech = speech_detector.process_frame(frame.copy())
    frame = processed_speech
    # We don't extract speech status yet (can be logged internally later)

    # --- Distance Estimation ---
    if face_landmarks:
        distance_results = distance_estimator.compute_distance(face_landmarks, frame.shape)
        distance_estimator.visualize(frame, distance_results)
        if distance_results:
            d, cat, *_ = distance_results[0]
            distance_cm = int(d)
            distance_cat = cat

    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log to CSV
    with open(log_path, mode="a", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            timestamp,
            gaze_dir,
            eyes_closed,
            tilt_dir,
            speech_status,
            distance_cm,
            distance_cat
        ])

    # Display frame
    cv2.imshow("Vision Trackker - All Modules", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
