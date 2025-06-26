import argparse

from app.vision.Track import Track
from app.vision.tilt_checker import run_tilt_monitor
from app.vision.speech_detector import run_speech_detection
from app.vision.distance_estimator import run_distance_monitor
from app.vision.gaze_tracker import run_gaze_tracker

def main():
    parser = argparse.ArgumentParser(description="Vision Trackker CLI")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["gaze", "speech", "distance", "tilt"],
                        help="Which module to run")

    args = parser.parse_args()

    if args.mode == "gaze":
        run_gaze_tracker()
    elif args.mode == "speech":
        run_speech_detection()
    elif args.mode == "distance":
        run_distance_monitor()
    elif args.mode == "tilt":
        run_tilt_monitor()
    else:
        print("Unknown mode selected.")

if __name__ == "__main__":
    main()
