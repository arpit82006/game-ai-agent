from __future__ import annotations
from dataclasses import dataclass
import time
import os
import inspect
from core.adb import Emulator


@dataclass
class TapEvent:
    """Immutable audit record of a physical touch input dispatch."""
    tap_id: int
    timestamp: float
    x: int
    y: int
    caller: str
    success: bool
    message: str


_GLOBAL_TAP_COUNTER: int = 0
_GLOBAL_TAP_LOG: list[TapEvent] = []


def get_total_physical_taps() -> int:
    """Return the total number of physical ADB taps dispatched in this process."""
    return _GLOBAL_TAP_COUNTER


def get_tap_history() -> list[TapEvent]:
    """Return a snapshot of all physical tap events recorded so far."""
    return list(_GLOBAL_TAP_LOG)


def clear_tap_history() -> None:
    """Reset the tap counter and history (used in unit testing)."""
    global _GLOBAL_TAP_COUNTER, _GLOBAL_TAP_LOG
    _GLOBAL_TAP_COUNTER = 0
    _GLOBAL_TAP_LOG.clear()


def tap(x: int, y: int, emulator: Emulator | None = None, purpose: str = "") -> tuple[bool, str]:
    """
    Send an ADB touch tap command to screen coordinates (X, Y) with complete audit logging.

    Args:
        x (int): Horizontal pixel coordinate on the emulator display.
        y (int): Vertical pixel coordinate on the emulator display.
        emulator (Emulator | None): Active emulator interface instance.
        purpose (str): Optional description of the tap intent (e.g. 'MOVE_SOURCE').

    Returns:
        tuple[bool, str]: (Success, output_or_error_message)
    """
    global _GLOBAL_TAP_COUNTER
    _GLOBAL_TAP_COUNTER += 1
    current_tap_id = _GLOBAL_TAP_COUNTER

    em = emulator or Emulator()
    caller_frame = inspect.stack()[1]
    filename = os.path.basename(caller_frame.filename)
    caller_info = f"{filename}:{caller_frame.lineno} in {caller_frame.function}"

    ts = time.time()
    ts_str = time.strftime('%H:%M:%S', time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"

    print(f"  [ADB TAP #{current_tap_id:03d}] {ts_str} -> ({x}, {y}) {purpose} (called by {caller_info})")

    try:
        out, err = em.run(["shell", "input", "tap", str(int(x)), str(int(y))])
        if err and "error" in err.lower():
            msg = f"ADB tap command failed at ({x}, {y}): {err}"
            _GLOBAL_TAP_LOG.append(TapEvent(current_tap_id, ts, x, y, caller_info, False, msg))
            return False, msg

        msg = f"Tapped ({x}, {y})"
        _GLOBAL_TAP_LOG.append(TapEvent(current_tap_id, ts, x, y, caller_info, True, msg))
        return True, msg
    except Exception as e:
        msg = f"ADB tap exception at ({x}, {y}): {e}"
        _GLOBAL_TAP_LOG.append(TapEvent(current_tap_id, ts, x, y, caller_info, False, msg))
        return False, msg
