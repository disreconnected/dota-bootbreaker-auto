"""State machine that turns a captured frame into a paddle action.

Launch is driven by the on-screen pre-throw prompt (the space-bar key icon),
not by guessing from ball-absence: we throw only when the game actually shows
"LOCK CART POSITION" / "THROW BOOT". States: AIM (prompt visible, no ball ->
tell main to launch), PLAYING (ball in play -> track), WAIT (neither -> hold).
"""

from bootbreaker.detect import detect_ball, detect_cart, detect_prethrow
from bootbreaker.strategy import decide_move, predict_intercept_x

# How many recent ball sightings to average velocity over. A single-frame
# velocity is far too noisy for the wall-bounce extrapolation (a few px of
# jitter, amplified by the look-ahead, throws the predicted landing hundreds of
# px off and drives the cart the wrong way). Averaging over a couple of frames
# smooths that out.
_VEL_WINDOW = 3

# The real ball is always moving. A detection pinned to the same spot for
# several frames is a static false positive - a boot-shaped brick in the wall or
# a UI icon that matches the cyan-glow + brown-boot signature. Reject it so the
# cart doesn't sit there oscillating under a phantom after the ball is lost.
_STATIC_EPS = 3  # px; movement at/below this counts as "not moving"
_STATIC_FRAMES = 5  # consecutive still frames -> treat as a static false positive


class Bot:
    def __init__(
        self,
        deadzone: float = 40,
        paddle_frac: float = 0.9,
        chase_gap: int = 6,
    ):
        self.deadzone = deadzone
        self.paddle_frac = paddle_frac
        self.chase_gap = chase_gap
        self.state = "WAIT"
        self._misses = 0
        self._history: list[tuple[int, int]] = []
        self._target_x = None
        self._last_pos = None  # raw last detection, for static-blob rejection
        self._static = 0  # consecutive frames the detection hasn't moved
        # Last detections, exposed for the debug view/logging.
        self.last_ball = None
        self.last_cart = None
        self.last_prethrow = False

    def _reset_tracking(self) -> None:
        self._history.clear()
        self._target_x = None

    def prime_playing(self) -> None:
        """Called right after a throw: reset tracking so the fresh ball is
        picked up cleanly."""
        self.state = "PLAYING"
        self._misses = 0
        self._reset_tracking()

    def step(self, frame) -> str:
        h, w = frame.shape[:2]
        cart = detect_cart(frame)
        self.last_cart = cart
        ball = detect_ball(frame, cart)
        self.last_ball = ball

        # --- reject static false positives ---
        # A detection that hasn't moved for a few frames isn't the ball (the
        # ball always moves); it's a fixed brick/icon. Drop it.
        if (
            ball is not None
            and self._last_pos is not None
            and abs(ball[0] - self._last_pos[0]) <= _STATIC_EPS
            and abs(ball[1] - self._last_pos[1]) <= _STATIC_EPS
        ):
            self._static += 1
        else:
            self._static = 0
        self._last_pos = ball
        if ball is not None and self._static >= _STATIC_FRAMES:
            ball = None
            self._target_x = None

        prethrow = detect_prethrow(frame)
        self.last_prethrow = prethrow

        # --- PLAYING: a ball is in play, track it ---
        if ball is not None:
            self.state = "PLAYING"
            self._misses = 0
            self._history.append(ball)
            if len(self._history) > _VEL_WINDOW:
                self._history.pop(0)
            # Follow the ball's x (pre-position); once it's descending, lead to
            # its predicted (wall-bounced) landing.
            if len(self._history) >= 2:
                x0, y0 = self._history[0]
                x1, y1 = self._history[-1]
                n = len(self._history) - 1
                vx = (x1 - x0) / n
                vy = (y1 - y0) / n
                paddle_y = cart[1] if cart is not None else int(self.paddle_frac * h)
                self._target_x = predict_intercept_x(ball[0], ball[1], vx, vy, paddle_y, w)
            else:
                self._target_x = ball[0]
            if cart is None or self._target_x is None:
                return "hold"
            return decide_move(self._target_x, cart[0], self.deadzone) or "hold"

        # --- no ball ---
        self._misses += 1
        self._history.clear()
        if prethrow:
            # The game is waiting for us to throw. This is the ONLY launch
            # trigger, so we never fire at a random moment.
            self.state = "AIM"
            self._reset_tracking()
            return "launch"

        # No ball and no prompt: a brief transition (the ball just left the cart
        # / a detection blip). Keep steering to the last target through the
        # blind-spot gap, then hold until the ball reappears or the prompt shows.
        self.state = "WAIT"
        if (
            self._target_x is not None
            and cart is not None
            and self._misses <= self.chase_gap
        ):
            return decide_move(self._target_x, cart[0], self.deadzone) or "hold"
        self._target_x = None
        return "hold"
