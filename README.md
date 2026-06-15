# 🎮 Subway Surfers Gesture Control

Control the **Subway Surfers** game using real-time hand gestures detected via webcam — no keyboard needed!

---

## 📌 Features

- Real-time hand detection using webcam
- Gesture recognition mapped to game controls
- No external hardware required — just a webcam
- Lightweight single-script implementation

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Python | Core programming language |
| MediaPipe | Real-time hand landmark detection |
| OpenCV | Webcam capture & frame processing |
| PyAutoGUI | Simulates keyboard inputs for game control |

---

## 🎯 Gesture Controls

| Hand Gesture | Game Action |
|---|---|
| Move hand left | Swipe Left |
| Move hand right | Swipe Right |
| Move hand up | Jump |
| Move hand down | Roll / Slide |

---

## 📁 Project Structure

```
subway-surfers-gesture-control/
└── project.py    # Main script — gesture detection & game control
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Webcam

### 1. Clone the repository
```bash
git clone https://github.com/VaishnaviYerramilli95/subway-surfers-gesture-control.git
cd subway-surfers-gesture-control
```

### 2. Install dependencies
```bash
pip install mediapipe opencv-python pyautogui
```

### 3. Open Subway Surfers
Open the game in your browser (BlueStacks or any emulator works too)

### 4. Run the script
```bash
python project.py
```

Point your webcam at your hand and start playing! 🤚

---

## 🧠 How It Works

1. OpenCV captures live webcam feed frame by frame
2. MediaPipe detects hand landmarks in each frame
3. Hand position is tracked and mapped to directional gestures
4. PyAutoGUI simulates the corresponding arrow key press
5. The game responds as if the keyboard was pressed

---

## 👩‍💻 Author

**Vaishnavi Yerramilli**
- GitHub: [@VaishnaviYerramilli95](https://github.com/VaishnaviYerramilli95)
- LinkedIn: [https://www.linkedin.com/in/vaishnavi-yerramilli995/]
