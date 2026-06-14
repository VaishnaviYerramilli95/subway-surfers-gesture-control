"""
game_controller.py
==================
Subway Surfers Gesture Control — Local Game Controller
NNDL Project

REQUIREMENTS (install separately, not needed on Render):
    pip install mediapipe opencv-python pyautogui

HOW TO USE:
    1. Open Subway Surfers in a browser window
    2. Click on the game window to give it focus
    3. In another terminal: python game_controller.py
    4. Show your hand gestures to the webcam!

GESTURES:
    ✋ Open Palm (raised)   → Jump  (↑)
    👊 Fist (lowered)       → Slide (↓)
    👈 Point/tilt left      → Left  (←)
    👉 Point/tilt right     → Right (→)
    🖐 Neutral              → No action
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# ─── Configuration ────────────────────────────────────────────────────────────
WEBCAM_INDEX    = 0          # Change if you have multiple cameras
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
CONFIDENCE_THR  = 0.65       # Minimum confidence to trigger a keypress
COOLDOWN_S      = 0.35       # Seconds between keypresses (prevents spam)
MODEL_PATH      = 'gesture_model.h5'
ENCODER_PATH    = 'label_encoder.pkl'

# Key mapping: gesture → pyautogui key name
KEY_MAP = {
    'jump':    'up',
    'slide':   'down',
    'left':    'left',
    'right':   'right',
    'neutral': None,
}

GESTURE_DISPLAY = {
    'jump':    ('JUMP  ↑', (150, 255, 150)),
    'slide':   ('SLIDE ↓', (150, 150, 255)),
    'left':    ('LEFT  ←', (255, 200, 100)),
    'right':   ('RIGHT →', (100, 200, 255)),
    'neutral': ('---',     (180, 180, 180)),
}


# ─── Load Model ───────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Run train_model.py first: python train_model.py"
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✅  Model loaded  ({model.count_params():,} parameters)")
    return model, label_encoder


# ─── Landmark Extraction ──────────────────────────────────────────────────────
def extract_landmarks(hand_landmarks):
    """Flatten 21 MediaPipe landmarks into a 63-element numpy array."""
    lm = []
    for landmark in hand_landmarks.landmark:
        lm += [landmark.x, landmark.y, landmark.z]
    return np.array(lm, dtype=np.float32).reshape(1, 63)


# ─── Overlay Drawing ──────────────────────────────────────────────────────────
def draw_overlay(frame, gesture, confidence, fps):
    """Draw gesture label, confidence bar, and FPS on the frame."""
    h, w = frame.shape[:2]
    label, color = GESTURE_DISPLAY.get(gesture, ('---', (180, 180, 180)))

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Gesture text
    cv2.putText(frame, label, (20, 48), cv2.FONT_HERSHEY_DUPLEX,
                1.4, color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_w = int((w - 300) * confidence)
    cv2.rectangle(frame, (260, 22), (w - 20, 48), (40, 40, 60), -1)
    cv2.rectangle(frame, (260, 22), (260 + bar_w, 48), color, -1)
    cv2.putText(frame, f'{confidence*100:.0f}%', (w - 70, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)

    # FPS
    cv2.putText(frame, f'FPS: {fps:.0f}', (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

    # Instructions
    cv2.putText(frame, 'Q = Quit', (w - 100, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    return frame


# ─── Main Loop ────────────────────────────────────────────────────────────────
def main():
    print("\n🎮  Subway Surfers Gesture Controller")
    print("=" * 45)

    model, label_encoder = load_model()

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"❌  Cannot open camera index {WEBCAM_INDEX}")
        return

    print("✅  Camera opened. Show your hand to control the game!")
    print("    Focus the game window, then come back here.")
    print("    Press Q in the camera window to quit.\n")

    last_keypress   = 0.0
    last_gesture    = 'neutral'
    fps_prev_time   = time.time()

    pyautogui.FAILSAFE = False   # Allow moving mouse to corner without abort

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌  Frame capture failed. Exiting.")
            break

        frame = cv2.flip(frame, 1)   # Mirror for natural feel
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture    = 'neutral'
        confidence = 1.0

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            features   = extract_landmarks(hand_lm)
            probs      = model.predict(features, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            gesture    = label_encoder[pred_idx]
            confidence = float(probs[pred_idx])

            # Trigger keypress with cooldown
            now = time.time()
            key = KEY_MAP.get(gesture)
            if (key is not None and confidence >= CONFIDENCE_THR
                    and now - last_keypress >= COOLDOWN_S):
                pyautogui.press(key)
                last_keypress = now
                print(f"  🎮  {gesture.upper():8s}  ({confidence*100:.0f}%)  → {key}")

        last_gesture = gesture

        # FPS calculation
        now     = time.time()
        fps     = 1.0 / max(now - fps_prev_time, 1e-6)
        fps_prev_time = now

        frame = draw_overlay(frame, gesture, confidence, fps)
        cv2.imshow('Subway Surfers — Gesture Control (Q to quit)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n👋  Controller stopped.")


if __name__ == '__main__':
    main()
