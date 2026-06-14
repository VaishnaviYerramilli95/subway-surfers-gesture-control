def detect_gesture(landmarks):
    left_hand = landmarks[15]
    right_hand = landmarks[16]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    # Both hands up → jump
    if left_hand.y < left_shoulder.y and right_hand.y < right_shoulder.y:
        return "jump"

    # Left hand up → move left (mirror fixed)
    elif left_hand.y < left_shoulder.y:
        return "right"

    # Right hand up → move right (mirror fixed)
    elif right_hand.y < right_shoulder.y:
        return "left"

    return None