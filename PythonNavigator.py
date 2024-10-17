import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera
cap = cv2.VideoCapture(0)

# Variables for gesture tracking
tap_start_time = 0
tap_duration = 0.2  # Duration for tap gesture (in seconds)
hold_duration = 0.5  # Duration for hold gesture (in seconds)
is_holding = False

# Get screen size
screen_width, screen_height = pyautogui.size()

# Sensitivity factor (increase for more sensitive movement)
sensitivity = 1.5

def detect_index_thumb_tap(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    distance = ((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)**0.5
    return distance < 0.04  # Adjusted threshold for more sensitive detection

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Move mouse pointer with index finger (increased sensitivity)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            mouse_x = int(screen_width / 2 + (index_tip.x - 0.5) * screen_width * sensitivity)
            mouse_y = int(screen_height / 2 + (index_tip.y - 0.5) * screen_height * sensitivity)
            mouse_x = max(0, min(mouse_x, screen_width - 1))
            mouse_y = max(0, min(mouse_y, screen_height - 1))
            pyautogui.moveTo(mouse_x, mouse_y)

            # Detect index finger and thumb tap (Left click and hold)
            if detect_index_thumb_tap(hand_landmarks):
                if tap_start_time == 0:
                    tap_start_time = time.time()
                elif time.time() - tap_start_time > tap_duration:
                    pyautogui.click()
                    tap_start_time = 0
                is_holding = True
            else:
                tap_start_time = 0
                is_holding = False

    cv2.imshow('Hand Gesture Navigation', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False
