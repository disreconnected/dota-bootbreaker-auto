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


def test_predicts_slow_far_ball_when_motion_is_trusted():
    # A steady slow boot should receive the same landing calculation as a fast
    # one: y=688 -> 956 at vy=6 takes 44.7 frames, landing at x=455.
    x = strategy.predict_intercept_x(187, 688, 6, 6, 956, 830)
    assert x == 455


def test_optional_prediction_horizon_remains_available_for_noisy_callers():
    x = strategy.predict_intercept_x(187, 688, 6, 6, 956, 830, max_lookahead=4)
    assert x == 187


def test_estimate_velocity_uses_median_for_a_slow_clean_trajectory():
    velocity = strategy.estimate_velocity([(100, 100), (106, 106), (112, 112), (118, 118)])
    assert velocity is not None
    vx, vy, confidence = velocity
    assert (vx, vy) == (6, 6)
    assert confidence == 1.0


def test_adaptive_deadzone_tightens_near_the_paddle():
    assert strategy.adaptive_deadzone(40, 4) == 40
    assert strategy.adaptive_deadzone(40, 0) == 12
