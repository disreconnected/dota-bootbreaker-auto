"""Pure decision logic: paddle movement and game state transitions."""


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


def predict_intercept_x(
    ball_x: float,
    ball_y: float,
    vx: float,
    vy: float,
    paddle_y: float,
    width: float,
    max_lookahead: float = 4.0,
) -> float:
    """Predict the ball's x when it descends to the paddle line, accounting for
    wall bounces. If the ball is not descending, just track its current x.

    ``max_lookahead`` caps how far ahead we trust the extrapolation. ``t`` is the
    number of frames of travel to reach the paddle; when it is large the ball is
    high and/or falling slowly, so a small ``vy`` blows the ``vx * t`` term up
    into garbage. In that case we don't predict - we just track the current x
    and re-decide once the ball is genuinely falling (small ``t``)."""
    if vy <= 0:
        return float(ball_x)
    t = (paddle_y - ball_y) / vy
    if t < 0 or t > max_lookahead:
        return float(ball_x)
    return _reflect(ball_x + vx * t, width)
