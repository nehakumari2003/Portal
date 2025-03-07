import cv2
import mediapipe as mp
import numpy as np
import time

class GestureCameraController:
    def __init__(self):
        # MediaPipe Hands Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Camera Initialization
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Cannot open camera")

        # State Variables
        self.camera_on = True
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.5  # Seconds between gestures

    def detect_gesture(self, hand_landmarks):
        """
        Precise thumbs up/down detection
        Requires:
        - Clear vertical hand orientation
        - Significant thumb position relative to other fingers
        """
        # Key landmarks
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Strict vertical alignment check
        vertical_alignment = (
            abs(thumb_tip.x - wrist.x) < 0.15 and
            abs(index_tip.x - wrist.x) < 0.15
        )

        # Vertical position difference
        thumb_index_diff = thumb_tip.y - index_tip.y
        thumb_middle_diff = thumb_tip.y - middle_tip.y

        # Precise gesture detection with multiple checks
        if vertical_alignment:
            if thumb_index_diff < -0.1 and thumb_middle_diff < -0.1:
                return "thumbs_up"
            elif thumb_index_diff > 0.1 and thumb_middle_diff > 0.1:
                return "thumbs_down"
        
        return None

    def run(self):
        print("Thumbs UP to turn ON, Thumbs DOWN to turn OFF")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            current_time = time.time()

            # Focus only on hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Only process gesture if cooldown passed
                    if current_time - self.last_gesture_time > self.gesture_cooldown:
                        gesture = self.detect_gesture(hand_landmarks)

                        if gesture == "thumbs_up" and not self.camera_on:
                            self.camera_on = True
                            self.last_gesture_time = current_time
                            print("Camera turned ON")

                        elif gesture == "thumbs_down" and self.camera_on:
                            self.camera_on = False
                            self.last_gesture_time = current_time
                            print("Camera turned OFF")

            # Display based on camera state
            if self.camera_on:
                cv2.putText(frame, "Camera: ON", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Gesture Camera Control", frame)
            else:
                blank_screen = np.zeros_like(frame)
                cv2.putText(blank_screen, "Camera: OFF", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Gesture Camera Control", blank_screen)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()

def main():
    try:
        controller = GestureCameraController()
        controller.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()