import pyautogui
import time

pyautogui.FAILSAFE = False

def perform_action(action):
    if action == "left":
        pyautogui.press('left')

    elif action == "right":
        pyautogui.press('right')

    elif action == "jump":
        pyautogui.press('space')

    elif action == "roll":
        pyautogui.press('down')