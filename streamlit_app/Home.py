import streamlit as st
import cv2
import time
import os
import csv
from datetime import datetime
import numpy as np
import mediapipe as mp

from app.vision.gaze_tracker import GazeTracker
from app.vision.tilt_checker import HeadTiltChecker
from app.vision.speech_detector import SpeechDetector
from app.vision.distance_estimator import DistanceEstimator

# ---------- Configuration ----------
st.set_page_config(page_title="Vision Trackker", layout="wide")
st.title("🎯 Vision Trackker — Unified Attention Monitoring")

# Setup module toggles
st.sidebar.header("Select Modules to Run")
use_gaze = st.sidebar.checkbox("Gaze Tracking", value=True)
use_tilt = st.sidebar.checkbox("Head Tilt Detection", value=True)
use_speech = st.sidebar.checkbox("Speech Detection", value=True)
use_distance = st.sidebar.checkbox("Distance Estimation", value=True)

# Webcam start button
start = st.sidebar.button("🚀 Start Vision Monitor")

# CSV Logging setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "session_log.csv")

# ---------- When 'Start' is pressed ----------
if start:
    stframe = st.empty()
    st.sidebar.success("Running... press stop to quit")

    # Initialize models
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    gaze_tracker = GazeTracker() if use_gaze else None
    tilt_checker = HeadTiltChecker() if use_tilt else None
    speech_detector = SpeechDetector() if use_speech else None
    distance_estimator = DistanceEstimator() if use_distance else None

    # Write CSV header if needed
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                "timestamp", "gaze_direction", "eyes_closed",
                "tilt_direction", "speech_status",
                "distance_cm", "distance_category"
            ])

    # Webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)
        landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None

        # Output placeholders
        gaze_dir, eyes_closed = "N/A", "N/A"
        tilt_dir = "N/A"
        speech_status = "N/A"
        distance_cm, distance_cat = "N/A", "N/A"

        # Run modules
        if use_gaze and landmarks:
            frame, gaze_dir, eyes_closed = gaze_tracker.process_frame(frame, landmarks)

        if use_tilt:
            frame = tilt_checker.process_frame(frame)

        if use_speech:
            frame = speech_detector.process_frame(frame)

        if use_distance and landmarks:
            dists = distance_estimator.compute_distance(landmarks, frame.shape)
            distance_estimator.visualize(frame, dists)
            if dists:
                d, cat, *_ = dists[0]
                distance_cm = int(d)
                distance_cat = cat

        # Logging
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                timestamp, gaze_dir, eyes_closed,
                tilt_dir, speech_status, distance_cm, distance_cat
            ])

        # Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Control frame rate
        time.sleep(0.03)

    cap.release()
    st.sidebar.warning("🛑 Stream stopped.")
