# 🧠 FocusLens  
### *"Track Focus. Respect Privacy."*

**FocusLens** is a real-time, privacy-first engagement tracker built for remote professionals. It uses lightweight computer vision and audio detection to measure presence, focus, and interaction — without intrusive screen recordings.

Whether you're managing a team, teaching online, or optimizing your own productivity, FocusLens gives you **insight without surveillance**.

---

## 🚀 Key Features

| Feature                       | Description                                                             |
| ----------------------------- | ----------------------------------------------------------------------- |
| 🎯 **Gaze Detection**         | Determines if the user is looking at the screen or distracted           |
| 📏 **Distance Estimation**    | Monitors how far the user is sitting from the screen (ergonomics/focus) |
| 🗣️ **Speech Activity**        | Detects when the user is speaking (meeting participation)               |
| 🧍 **Head Tilt Monitoring**   | Measures posture cues (e.g., looking down, turning away)                |
| ⏱️ **Engagement Session Log** | Automatically tracks session start/end and logs engagement duration     |
| 🖥️ **Webcam-Only Operation**  | No screen or mic recording — only non-intrusive face-based insights     |
| 📊 **Streamlit Frontend**     | Live dashboard showing indicators and webcam feed in real time          |
| 📁 **Unified CSV Logging**    | Outputs structured logs for further analysis or reporting               |

---

## 👤 Ideal For

* 👥 Remote teams & project managers  
* 🎓 Online instructors & trainers  
* 🧠 Focus-conscious freelancers  
* 🧩 Researchers studying human attention  
* 🧰 Productivity app developers

---

## 🛠️ Tech Stack

* **Python 3.8+**
* `OpenCV`, `MediaPipe`, `NumPy`, `FilterPy`
* `Streamlit` (for frontend)
* Minimal dependencies, runs locally (no cloud processing required)

---

## 🧱 Project Structure

```bash
vision-trackker/
├── app.py                  # Run all modules together (logs + webcam)
├── run.py                  # CLI entry to individual modules
├── requirements.txt
├── .gitignore
├── logs/                   # Engagement session CSV logs
│   └── session_log.csv
├── app/
│   └── vision/
│       ├── gaze_tracker.py
│       ├── distance_estimator.py
│       ├── speech_detector.py
│       ├── tilt_checker.py
├── tracking/
│   └── sort.py             # Object tracking (if needed)
└── streamlit_app/
    └── Home.py             # Streamlit-based frontend
````

---

## 🧪 Getting Started

### 🔧 Install Requirements

```bash
pip install -r requirements.txt
```

### ▶️ Run Full Tracker

```bash
python app.py
```

### 🖥️ Launch Web UI

```bash
streamlit run streamlit_app/Home.py
```

---

## 📦 Output Format

Every engagement session gets logged in a structured CSV like:

```csv
timestamp,gaze_direction,eyes_closed,tilt_direction,speech_status,distance_cm,distance_category
2025-06-26 15:00:12,Center,False,Up,Speaking,48,Mid-range
```

*Note: Values shown are examples. Actual output depends on live detection accuracy and data availability per frame.*

---

## 🔐 Privacy First

FocusLens **never records your screen, microphone, or saves images/video**.
It only uses visual metadata (face landmarks, gaze angles) processed **locally and in real-time**, then discarded.

No data leaves your device.

---

## 📈 What’s Next

* 🧭 Real-time visual engagement charts
* 🌍 Multi-user meeting summary logs
* 📦 Export to calendar/time-tracking tools
* 🧠 AI-generated engagement scores

---

## 🚧 Work in Progress

These features are in development. Contributions and feedback welcome!
