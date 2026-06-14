import cv2
import mediapipe as mp
import threading
import time
import voice_control

from gesture_control import detect_gesture
from key_control import perform_action

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

paused = False
last_action_time = 0

# Start voice thread
threading.Thread(target=voice_control.listen_command, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    action = None

    # Gesture detection
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        action = detect_gesture(landmarks)

    # Voice override
    if voice_control.voice_command:
        action = voice_control.voice_command
        voice_control.voice_command = ""

    # Execute action
    if action and time.time() - last_action_time > 1:

        print("Action:", action)

        if action == "stop":
            break

        elif action == "pause":
            paused = True

        elif action == "resume":
            paused = False

        else:
            if not paused:
                perform_action(action)

        last_action_time = time.time()

    # Draw pose
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    status = "PAUSED" if paused else "RUNNING"

    cv2.putText(frame, f"Status: {status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Action: {action}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()