import cv2
import mediapipe as mp
import time

class HeadTiltChecker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.session_active = False
        self.session_start_time = None

    def process_frame(self, img):
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        frame_center = (w // 2, h // 2)

        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                # Find bounding box
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in face.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                box_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

                # Offsets
                dx = box_center[0] - frame_center[0]
                dy = box_center[1] - frame_center[1]
                percent_x = (dx / (w / 2)) * 100
                percent_y = (dy / (h / 2)) * 100

                horiz = "Right" if percent_x > 0 else "Left"
                vert = "Down" if percent_y > 0 else "Up"

                # Draw bounding box and center
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                cv2.circle(img, box_center, 5, (0, 255, 0), -1)

                # Info Text
                cv2.putText(img, f"Horizontal: {horiz} ({abs(percent_x):.1f}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, f"Vertical: {vert} ({abs(percent_y):.1f}%)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if not self.session_active:
                self.session_active = True
                self.session_start_time = time.time()
                print(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.session_start_time))}")

        else:
            if self.session_active:
                self.session_active = False
                session_end = time.time()
                duration = session_end - self.session_start_time
                print(f"Session ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_end))}")
                print(f"Session duration: {duration:.2f} seconds")

        # Draw frame center
        cv2.circle(img, frame_center, 5, (0, 0, 255), -1)
        cv2.putText(img, "Frame Center (Origin)", 
                    (frame_center[0] - 100, frame_center[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img


def run_tilt_monitor():
    cap = cv2.VideoCapture(0)
    checker = HeadTiltChecker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        processed = checker.process_frame(frame)
        cv2.imshow("Head Tilt Monitor", processed)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
