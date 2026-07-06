"""Keyboard output for the game via pydirectinput (scancode SendInput)."""

import time

import pydirectinput

# pydirectinput sleeps this many seconds after EVERY key event (default ~0.1s).
# That stalled the whole capture loop to ~7 fps on every direction change -
# exactly when the cart needs to react - so the ball jumped 100+px and detection
# dropped mid-catch. We don't need any inter-key delay; kill it for full speed.
pydirectinput.PAUSE = 0

# How long tap_space holds space down. pydirectinput.press() sends keyDown then
# keyUp back-to-back; with PAUSE=0 (above) that tap lasts microseconds, which
# Dota's per-frame (~16ms) input polling misses entirely - the cart never
# locked/threw. So we hold space explicitly for a few frames. This only runs in
# _launch (which already sleeps hundreds of ms), never in the tracking loop.
_TAP_HOLD = 0.06


class Controller:
    def __init__(self, backend=pydirectinput):
        self._backend = backend
        self._held: set[str] = set()

    def hold(self, key: str) -> None:
        if key in self._held:
            return
        for other in list(self._held):
            self._backend.keyUp(other)
            self._held.discard(other)
        self._backend.keyDown(key)
        self._held.add(key)

    def release_all(self) -> None:
        for key in list(self._held):
            self._backend.keyUp(key)
            self._held.discard(key)

    def tap_space(self) -> None:
        # Hold space long enough for the game to register it (see _TAP_HOLD).
        self._backend.keyDown("space")
        time.sleep(_TAP_HOLD)
        self._backend.keyUp("space")
