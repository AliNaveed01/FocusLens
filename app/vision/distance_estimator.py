import cv2
import mediapipe as mp
import time

class DistanceEstimator:
    """
    Estimates distance of user's face from camera using forehead-to-chin ratio.
    Categorizes proximity and logs session duration based on distance.
    """
    def __init__(self, real_face_height=20):
        """
        :param real_face_height: Average real-world height of a human face in cm.
        """
        self.real_face_height = real_face_height
        self.session_active = False
        self.session_start_time = None

    def compute_distance(self, face_landmarks, frame_shape):
        """
        Computes face distance using forehead-chin vertical span.
        :param face_landmarks: Mediapipe face landmarks
        :param frame_shape: (H, W, C) shape of frame
        :return: List of distance results per face
        """
        if not face_landmarks:
            if self.session_active:
                self.end_session()
            return []

        h, w = frame_shape[:2]
        # Mediapipe landmark index for forehead and chin
        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]

        # Convert to pixel coordinates
        forehead_coords = (int(forehead.x * w), int(forehead.y * h))
        chin_coords = (int(chin.x * w), int(chin.y * h))

        # Pixel height of face
        face_height_pixels = abs(forehead_coords[1] - chin_coords[1])
        if face_height_pixels > 0:
            # Estimate distance based on face size in frame
            distance = (w * self.real_face_height) / face_height_pixels
            category = self.get_distance_category(distance)

            # Trigger session start/end based on proximity
            if category not in ["Far", "Too Far"]:
                self.start_session()
            else:
                self.end_session()

            return [(distance, category, forehead_coords, chin_coords)]

        if self.session_active:
            self.end_session()
        return []

    def visualize(self, frame, results):
        """
        Annotates distance and category info on frame.
        :param frame: Input frame
        :param results: Output of compute_distance
        """
        for distance, category, forehead_coords, chin_coords in results:
            cv2.circle(frame, forehead_coords, 5, (0, 255, 0), -1)
            cv2.circle(frame, chin_coords, 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Distance: {int(distance)}',
                        (forehead_coords[0] + 10, forehead_coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f'Category: {category}',
                        (forehead_coords[0] + 10, forehead_coords[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def get_distance_category(self, distance):
        """
        Categorizes a numerical distance into labels.
        """
        if distance < 20:
            return "Immediate"
        elif 20 <= distance < 45:
            return "Near"
        elif 45 <= distance < 90:
            return "Mid-range"
        elif 90 <= distance < 180:
            return "Far"
        else:
            return "Too Far"

    def start_session(self):
        """
        Starts a session if a user is close enough.
        """
        if not self.session_active:
            self.session_start_time = time.time()
            self.session_active = True
            print(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.session_start_time))}")

    def end_session(self):
        """
        Ends the session and logs the duration.
        """
        if self.session_active:
            session_end_time = time.time()
            duration = session_end_time - self.session_start_time
            print(f"Session ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_end_time))}")
            print(f"Session duration: {duration:.2f} seconds")
            self.session_active = False


def run_distance_monitor():
    """
    Live demo runner using webcam.
    """
    cap = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    estimator = DistanceEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)
        face_landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None

        distances = estimator.compute_distance(face_landmarks, frame.shape)
        estimator.visualize(frame, distances)

        cv2.imshow('Distance Estimation', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
