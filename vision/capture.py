import pygetwindow as gw
import pyautogui
import os
import time

# Give you 2 seconds to bring BlueStacks to the front
print("Switch to BlueStacks...")
time.sleep(2)

# Find the BlueStacks window
windows = gw.getWindowsWithTitle("BlueStacks App Player")

if not windows:
    print("BlueStacks window not found!")
    exit()

window = windows[0]

# Bring it to the front
window.activate()
time.sleep(1)

# Get window position and size
left = window.left
top = window.top
width = window.width
height = window.height

print(f"Window Position: ({left}, {top})")
print(f"Window Size: {width} x {height}")

# Create screenshots folder if needed
os.makedirs("screenshots", exist_ok=True)

# Take screenshot
image = pyautogui.screenshot(
    region=(left, top, width, height)
)

save_path = "screenshots/test_capture.png"
image.save(save_path)

print(f"Screenshot saved to: {save_path}")