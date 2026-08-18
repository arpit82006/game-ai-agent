import subprocess
import os
import time
import cv2

ADB_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
DEVICE = "127.0.0.1:5555"


class Emulator:
    """
    Interface to BlueStacks Android instance via HD-Adb.
    """

    def __init__(self, device=None, adb_path=None):
        self.adb_path = adb_path or ADB_PATH
        self.device = device or DEVICE

    def check_adb_executable(self) -> bool:
        """Verify that HD-Adb.exe exists on disk."""
        return os.path.exists(self.adb_path)

    def connect(self) -> tuple[bool, str]:
        """
        Ensure ADB daemon is running and connected to BlueStacks.

        Returns:
            tuple[bool, str]: (Success, status_or_error_message)
        """
        if not self.check_adb_executable():
            return False, f"HD-Adb.exe not found at: {self.adb_path}"

        # Try connecting to primary IP
        subprocess.run([self.adb_path, "connect", self.device], capture_output=True, text=True)

        # Check attached devices
        res = subprocess.run([self.adb_path, "devices"], capture_output=True, text=True)
        lines = [l.strip() for l in res.stdout.strip().split("\n")[1:] if l.strip()]

        active_devices = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                active_devices.append(parts[0])

        if not active_devices:
            return False, "No active BlueStacks emulator device found via ADB."

        # If primary device is in active list, use it; otherwise use first available active device
        if self.device in active_devices:
            return True, f"Connected to {self.device}"
        else:
            self.device = active_devices[0]
            return True, f"Connected to active device {self.device}"

    def check_bluestacks_running(self) -> bool:
        """Check if BlueStacks device is responding via ADB."""
        success, _ = self.connect()
        return success

    def run(self, command):
        """Run an ADB command targeting the active emulator device."""
        full_command = [
            self.adb_path,
            "-s",
            self.device
        ] + command

        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True
        )

        # Fallback if specific IP fails but emulator is attached
        if "device not found" in result.stderr.lower() or "error" in result.stderr.lower():
            dev_check = subprocess.run([self.adb_path, "devices"], capture_output=True, text=True)
            lines = [l.split()[0] for l in dev_check.stdout.strip().split("\n")[1:] if l.strip() and "device" in l]
            if lines and lines[0] != self.device:
                self.device = lines[0]
                full_command = [self.adb_path, "-s", self.device] + command
                result = subprocess.run(full_command, capture_output=True, text=True)

        return result.stdout.strip(), result.stderr.strip()

    def capture_screenshot(self, save_path="screenshots/screen.png") -> tuple[object, str]:
        """
        Capture a guaranteed fresh screenshot directly from BlueStacks.

        Uses a unique remote timestamped filename to prevent any emulator-side caching.

        Args:
            save_path (str): Destination local path for the screenshot.

        Returns:
            tuple[np.ndarray | None, str]: (Decoded BGR cv2 image or None, local file path)
        """
        connected, msg = self.connect()
        if not connected:
            raise ConnectionError(f"ADB connection failed: {msg}")

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        ts = int(time.time() * 1000)
        remote_file = f"/sdcard/screen_{ts}.png"

        # 1. Capture on Android
        out, err = self.run(["shell", "screencap", "-p", remote_file])
        if err and "error" in err.lower():
            raise RuntimeError(f"screencap failed: {err}")

        # 2. Pull to host
        out, err = self.run(["pull", remote_file, save_path])
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            raise RuntimeError(f"ADB pull failed or screenshot is empty: {err}")

        # 3. Clean up remote file
        self.run(["shell", "rm", remote_file])

        # 4. Load with OpenCV
        image = cv2.imread(save_path)
        if image is None:
            raise ValueError(f"Failed to decode captured screenshot from {save_path}")

        return image, save_path

    def screenshot(self):
        """Legacy screenshot capture for backward compatibility."""
        self.capture_screenshot("screenshots/screen.png")
        print("✅ Screenshot Saved!")


if __name__ == "__main__":
    emulator = Emulator()
    emulator.screenshot()