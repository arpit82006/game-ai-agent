import subprocess
import os

ADB_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
DEVICE = "127.0.0.1:5555"


class Emulator:

    def run(self, command):
        full_command = [
            ADB_PATH,
            "-s",
            DEVICE
        ] + command

        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True
        )

        return result.stdout.strip(), result.stderr.strip()

    def screenshot(self):

        os.makedirs("screenshots", exist_ok=True)

        self.run([
            "shell",
            "screencap",
            "-p",
            "/sdcard/screen.png"
        ])

        self.run([
            "pull",
            "/sdcard/screen.png",
            "screenshots/screen.png"
        ])

        print("✅ Screenshot Saved!")


if __name__ == "__main__":

    emulator = Emulator()

    emulator.screenshot()