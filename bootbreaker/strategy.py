"""Pure decision logic: paddle movement, motion estimation, and aiming."""

from __future__ import annotations

from statistics import median
from typing import Sequence


def decide_move(ball_x: float, cart_x: float, deadzone: float) -> str | None:
    delta = ball_x - cart_x
    if delta < -deadzone:
        return "left"
    if delta > deadzone:
        return "right"
    return None


def _reflect(x: float, width: float) -> float:
    """Fold x into [0, width], bouncing off the walls (triangle wave)."""
    if width <= 0:
        return x
    period = 2 * width
    x = x % period
    if x < 0:
        x += period
    return x if x <= width else period - x


def estimate_velocity(
    positions: Sequence[tuple[int, int]],
) -> tuple[float, float, float] | None:
    """Return robust ``(vx, vy, confidence)`` from recent frame positions.

    A median of frame-to-frame deltas keeps one blurred/late capture from
    sending the cart to the wrong wall.  The confidence score is deliberately
    based on how consistently the boot moves, not on its absolute speed: a
    slow, clean trajectory is as useful as a fast one.
    """
    if len(positions) < 2:
        return None

    deltas = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(positions, positions[1:])
    ]
    vx = float(median(dx for dx, _ in deltas))
    vy = float(median(dy for _, dy in deltas))
    speed = max(abs(vx), abs(vy))
    if speed < 1.0:
        return None

    # With one interval there is no consistency data yet, but a fresh,
    # visibly-moving boot is still worth a modest prediction.  More samples
    # quickly raise confidence when the motion is stable.
    if len(deltas) == 1:
        return vx, vy, 0.60

    residuals = [max(abs(dx - vx), abs(dy - vy)) for dx, dy in deltas]
    jitter = float(median(residuals))
    confidence = max(0.0, min(1.0, 1.0 - jitter / max(4.0, speed)))
    return vx, vy, confidence


def adaptive_deadzone(base: float, frames_to_paddle: float | None) -> float:
    """Tighten the control dead zone as an incoming boot gets close.

    Far from the cart, a wider band avoids useless left/right chatter.  Near a
    catch we reduce it to 30% of the configured value (never below 10 px), so
    the controller can still correct a fast or slow late descent.
    """
    if frames_to_paddle is None or frames_to_paddle >= 4.0:
        return float(base)
    urgency = max(0.0, min(1.0, frames_to_paddle / 4.0))
    return max(10.0, float(base) * (0.30 + 0.70 * urgency))


def predict_intercept_x(
    ball_x: float,
    ball_y: float,
    vx: float,
    vy: float,
    paddle_y: float,
    width: float,
    max_lookahead: float | None = None,
) -> float:
    """Predict the ball's x when it descends to the paddle line, accounting for
    wall bounces. If the ball is not descending, just track its current x.

    By default there is no speed-dependent horizon: a stable slow boot should
    receive the same wall-bounce prediction as a fast one.  Callers that have
    low-confidence motion data may still supply ``max_lookahead`` as a
    conservative fallback.
    """
    if vy <= 0:
        return float(ball_x)
    t = (paddle_y - ball_y) / vy
    if t < 0 or (max_lookahead is not None and t > max_lookahead):
        return float(ball_x)
    return _reflect(ball_x + vx * t, width)
