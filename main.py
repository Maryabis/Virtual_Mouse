import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
index_x, index_y = 0, 0

while True:
   
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        continue  

    frame = cv2.flip(frame, 1)  
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            index_finger = landmarks[8] 
            index_x = int(index_finger.x * frame_width)
            index_y = int(index_finger.y * frame_height)

            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), -1)  # Draw yellow circle on index tip

            thumb = landmarks[4] 
            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)

            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), -1)  # Draw yellow circle on thumb tip

            screen_x = int(index_finger.x * screen_width)
            screen_y = int(index_finger.y * screen_height)

            pyautogui.moveTo(screen_x, screen_y, duration=0.1)  

            if abs(index_y - thumb_y) < 20:
                pyautogui.click()
                pyautogui.sleep(0.5)  

    cv2.imshow('Virtual Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
