import os
import cv2
import mediapipe as mp
import datetime
import time
import threading
import mss  # Faster screenshot capture, install with: pip install mss

class OpenPalmScreenshotCapture:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Screenshot directory
        self.pictures_folder =  r"C:\Users\NEHA KUMARI\Pictures"


        # Ensure the directory exists
        if not os.path.exists(self.pictures_folder):
            os.makedirs(self.pictures_folder)

        # Palm detection variables
        self.palm_start_time = None
        self.palm_detection_min_time = 3  # Minimum time for open palm
        self.screenshot_taken = False
        self.message = ""
        self.message_timer = 0

        # Multi-threaded hand detection
        self.frame = None
        self.result = None
        self.processing = False

    def is_open_palm(self, hand_landmarks):
        """
        Detect if the hand is an open palm facing forward.
        """
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]

        # Check if the palm is facing forward
        is_forward_palm = (
            abs(wrist.x - thumb_tip.x) < 0.2 and  # Thumb close to wrist horizontally
            all(tip.y < wrist.y for tip in finger_tips)  # All finger tips above wrist
        )
        return is_forward_palm

    def process_frame(self):
        """
        Run Mediapipe Hand Detection in a separate thread.
        """
        while True:
            if self.frame is not None and not self.processing:
                self.processing = True
                rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.result = self.hands.process(rgb_frame)
                self.processing = False

    def take_screenshot(self):
        """
        Capture a screenshot using the mss library (faster than PyAutoGUI).
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = os.path.join(self.pictures_folder, f"screenshot_{timestamp}.png")

        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)

        print(f"Screenshot taken: {screenshot_path}")
        return screenshot_path

    def run(self):
        cap = cv2.VideoCapture(0)

        # Set lower resolution to improve speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Open Palm Screenshot Capture")
        print("Show open palm facing camera for 3 seconds to take a screenshot")

        # Start processing thread
        threading.Thread(target=self.process_frame, daemon=True).start()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            self.frame = cv2.flip(frame, 1)  # Flip image horizontally

            current_time = time.time()

            if self.result and self.result.multi_hand_landmarks:
                for hand_landmarks in self.result.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    if self.is_open_palm(hand_landmarks):
                        if self.palm_start_time is None:
                            self.palm_start_time = current_time

                        palm_duration = current_time - self.palm_start_time

                        if palm_duration >= self.palm_detection_min_time and not self.screenshot_taken:
                            # Take a screenshot
                            self.take_screenshot()
                            self.screenshot_taken = True
                            self.message = "Screenshot Taken!"
                            self.message_timer = 30
                    else:
                        self.palm_start_time = None
                        self.screenshot_taken = False
            else:
                self.palm_start_time = None
                self.screenshot_taken = False

            # Show message on the screen
            if self.message_timer > 0:
                cv2.putText(
                    self.frame, self.message, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                self.message_timer -= 1

            # Display the frame
            cv2.imshow("Open Palm Screenshot Capture", self.frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        screenshot_app = OpenPalmScreenshotCapture()
        screenshot_app.run()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()