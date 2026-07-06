from bootbreaker import strategy


def test_move_left_when_ball_left_of_cart():
    assert strategy.decide_move(ball_x=100, cart_x=200, deadzone=10) == "left"


def test_move_right_when_ball_right_of_cart():
    assert strategy.decide_move(ball_x=300, cart_x=200, deadzone=10) == "right"


def test_hold_when_within_deadzone():
    assert strategy.decide_move(ball_x=205, cart_x=200, deadzone=10) is None


def test_predict_upward_ball_returns_current_x():
    # Ball moving up (vy <= 0) -> just track current x.
    assert strategy.predict_intercept_x(100, 500, 5, -10, 900, 830) == 100


def test_predict_straight_down_returns_ball_x():
    assert strategy.predict_intercept_x(400, 100, 0, 200, 900, 830) == 400


def test_predict_leads_descending_ball():
    # ball at x=120,y=300 moving (20,200); paddle at y=1000 -> t=3.5 -> x=190.
    assert strategy.predict_intercept_x(120, 300, 20, 200, 1000, 830) == 190


def test_predict_reflects_off_right_wall():
    # Unbounced landing would be 1000 in a 830-wide field -> reflect to 660.
    x = strategy.predict_intercept_x(600, 0, 200, 200, 400, 830)
    assert x == 660  # ball_x + vx*t = 600+400=1000 -> 2*830-1000 = 660


def test_predict_ignores_slow_far_ball_and_tracks_x():
    # Real failure from a play log: ball high (y=688) barely descending (vy=6)
    # with the paddle at y=956 -> t=44.7. The old code extrapolated vx*t into
    # garbage (predicted x=455 while the ball was heading to x~50, driving the
    # cart the wrong way). With the look-ahead guard it just tracks the ball's x.
    x = strategy.predict_intercept_x(187, 688, 6, 6, 956, 830)
    assert x == 187
