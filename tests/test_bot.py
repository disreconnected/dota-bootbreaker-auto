import numpy as np

import bootbreaker.bot as bot_mod
from bootbreaker.bot import Bot
from bootbreaker.detect import BounceSurface, PaddleSurface, SpecialTarget

FRAME = np.zeros((1080, 830, 3), dtype="uint8")  # dummy; detect is patched


def _seq(monkeypatch, balls, cart, prethrow=False):
    it = iter(balls)
    monkeypatch.setattr(bot_mod, "detect_ball", lambda frame, cart=None: next(it))
    monkeypatch.setattr(bot_mod, "detect_cart", lambda frame: cart)
    monkeypatch.setattr(
        bot_mod,
        "detect_paddle_surface",
        lambda frame, detected_cart=None: (
            PaddleSurface(cart[0] - 80, cart[0] + 80, cart[1] - 10)
            if cart is not None
            else None
        ),
    )
    monkeypatch.setattr(
        bot_mod, "detect_prethrow", lambda frame, threshold=None: prethrow
    )


def test_tracks_from_first_sighting(monkeypatch):
    # Pre-position immediately: on first sighting, steer toward the ball's x.
    _seq(monkeypatch, [(200, 100), (200, 300)], (400, 1000))
    b = Bot(deadzone=20)
    assert b.step(FRAME) == "left"  # target 200 < cart 400
    assert b.state == "PLAYING"


def test_moves_right_on_descending_ball(monkeypatch):
    _seq(monkeypatch, [(600, 100), (600, 300)], (400, 1000))
    b = Bot(deadzone=20)
    b.step(FRAME)
    assert b.step(FRAME) == "right"


def test_uses_a_safe_combo_bias_on_a_straight_descent(monkeypatch):
    _seq(monkeypatch, [(410, 100), (410, 300)], (400, 1000))
    b = Bot(deadzone=20)
    b.step(FRAME)
    # A centred incoming boot would normally hold. Combo mode shifts the cart
    # slightly left so the rebound leaves right and can build airborne bounces.
    assert b.step(FRAME) == "left"


def test_tracks_under_ascending_ball(monkeypatch):
    # Even while the ball rises, keep the cart under it (pre-position for the
    # descent) rather than sitting still and falling behind.
    _seq(monkeypatch, [(200, 400), (200, 200)], (400, 1000))
    b = Bot(deadzone=20)
    b.step(FRAME)
    assert b.step(FRAME) == "left"  # track toward ball x=200 (< cart 400)


def test_holds_when_cart_missing(monkeypatch):
    _seq(monkeypatch, [(100, 100), (100, 300)], None)
    b = Bot(deadzone=20)
    b.step(FRAME)
    assert b.step(FRAME) == "hold"


def test_leads_descending_ball(monkeypatch):
    _seq(monkeypatch, [(100, 100), (120, 300)], (100, 1000))
    b = Bot(deadzone=20)
    b.step(FRAME)
    # vx=20, vy=200, paddle_y=1000 -> t=3.5 -> x=190 > cart 100 -> right
    assert b.step(FRAME) == "right"


def test_keeps_moving_when_ball_briefly_lost(monkeypatch):
    _seq(monkeypatch, [(500, 100), (500, 300), None], (100, 1000))
    b = Bot(deadzone=20)
    b.step(FRAME)  # first sighting -> hold
    assert b.step(FRAME) == "right"  # descending, target 500 -> right
    assert b.step(FRAME) == "right"  # ball lost but keep target 500 -> right


def test_stops_chasing_stale_target_after_long_gap(monkeypatch):
    # Short gaps keep steering (a catch briefly hides the ball); a long absence
    # means it left play -> stop thrashing the stale target.
    _seq(monkeypatch, [(700, 300), (700, 500)] + [None] * 7, (400, 1000))
    b = Bot(deadzone=20, chase_gap=6)
    b.step(FRAME)  # first sighting -> target 700
    b.step(FRAME)  # descending -> target 700
    results = [b.step(FRAME) for _ in range(7)]  # misses 1..7
    assert results[5] == "right"  # miss 6: still within chase_gap
    assert results[6] == "hold"   # miss 7: gone too long -> stop


def test_ignores_static_false_positive(monkeypatch):
    # A fixed screen feature (boot-shaped brick / UI icon) detected at the exact
    # same spot every frame is NOT the ball. After a few still frames the bot
    # must stop chasing it and hold, instead of oscillating under the phantom.
    _seq(monkeypatch, [(415, 63)] * 8, (600, 1000))
    b = Bot(deadzone=20, chase_gap=6)
    for _ in range(5):
        b.step(FRAME)  # can't tell it's static yet -> tracks it
    assert b.step(FRAME) == "hold"  # recognized as static -> hold
    assert b.step(FRAME) == "hold"  # stays held while the phantom persists


def test_launches_only_when_prethrow_prompt_visible(monkeypatch):
    # No ball + the ␣ prompt on screen -> launch.
    _seq(monkeypatch, [None], (400, 1000), prethrow=True)
    b = Bot()
    assert b.step(FRAME) == "launch"
    assert b.state == "AIM"


def test_no_launch_without_prompt_even_when_ball_gone(monkeypatch):
    # Ball gone but no prompt (a transition / detection blip): must NOT launch -
    # the old ball-absence heuristic used to fire at random moments here.
    _seq(monkeypatch, [None] * 12, (400, 1000), prethrow=False)
    b = Bot()
    results = [b.step(FRAME) for _ in range(12)]
    assert "launch" not in results


def test_prime_playing_resets_to_playing(monkeypatch):
    _seq(monkeypatch, [None], (100, 1000), prethrow=False)
    b = Bot()
    b.prime_playing()
    assert b.state == "PLAYING"
    assert b.step(FRAME) != "launch"  # no prompt -> no relaunch


def test_one_up_mission_outranks_gold(monkeypatch):
    _seq(monkeypatch, [(300, 100)], (400, 1000))
    monkeypatch.setattr(
        bot_mod,
        "detect_special_targets",
        lambda frame: [SpecialTarget("gold", 280, 200), SpecialTarget("one_up", 620, 150)],
    )

    b = Bot()
    b.step(FRAME)

    assert b.mission is not None
    assert b.mission.kind == "one_up"
    assert b.mission_mode == "one_up"


def test_bonus_mission_nudges_a_safe_catch_toward_target(monkeypatch):
    _seq(monkeypatch, [(400, 100), (400, 300)], (430, 1000))
    monkeypatch.setattr(
        bot_mod,
        "detect_special_targets",
        lambda frame: [SpecialTarget("gold", 700, 200)],
    )

    b = Bot(deadzone=40)
    assert b.step(FRAME) == "hold"
    # The normal catch target is x=400. The gold block is to the right, so the
    # cart deliberately shifts left by a small, catch-safe amount.
    assert b.step(FRAME) == "left"
    assert b.last_intercept == 400
    assert b._target_x < b.last_intercept


def test_unproductive_one_up_mission_switches_to_advance_after_attempt_budget(monkeypatch):
    _seq(monkeypatch, [(300, 100), (300, 200)], (400, 1000))
    monkeypatch.setattr(
        bot_mod,
        "detect_special_targets",
        lambda frame: [SpecialTarget("one_up", 300, 200)],
    )
    b = Bot()

    b.step(FRAME)
    assert b.mission_mode == "one_up"
    b._mission_attempts = 5
    b.step(FRAME)

    assert b.mission is None
    assert b.mission_mode == "advance"


def test_gold_retargets_after_three_return_opportunities(monkeypatch):
    _seq(monkeypatch, [(200, 100), (200, 200)], (400, 1000))
    monkeypatch.setattr(
        bot_mod,
        "detect_special_targets",
        lambda frame: [SpecialTarget("gold", 200, 200), SpecialTarget("gold", 620, 200)],
    )
    b = Bot()

    b.step(FRAME)
    assert b.mission == SpecialTarget("gold", 200, 200)
    b._mission_attempts = 3
    b.step(FRAME)

    assert b.mission == SpecialTarget("gold", 620, 200)
    assert b.mission_mode == "gold"


def test_gold_retargets_on_the_first_confirmed_board_progress(monkeypatch):
    _seq(monkeypatch, [(200, 100), (200, 200)], (400, 1000))
    gold_a = SpecialTarget("gold", 200, 200)
    gold_b = SpecialTarget("gold", 620, 200)
    monkeypatch.setattr(bot_mod, "detect_special_targets", lambda frame: [gold_a, gold_b])
    b = Bot()

    b.step(FRAME)
    assert b.mission == gold_a
    # This flag is set by the board-mass progress observer when the aimed gold
    # is destroyed; it must take effect on the very next frame.
    b._force_retarget_gold = True
    b.step(FRAME)

    assert b.mission == gold_b


def test_no_rail_means_no_risky_combo_or_gold_offset():
    b = Bot()

    assert b._aim_for_combo(400, 830, max_offset=0) == 400
    assert b._aim_for_target(
        400, SpecialTarget("gold", 650, 200), 830, max_offset=0, landing_x=400
    ) == 400


def test_board_redraw_without_boot_or_prompt_requests_recalibration(monkeypatch):
    _seq(monkeypatch, [None] * 24, (400, 1000), prethrow=False)
    monkeypatch.setattr(bot_mod, "estimate_breakable_mass", lambda frame: 1_300)
    b = Bot()
    b._last_live_board_mass = 1_000

    actions = [b.step(FRAME) for _ in range(24)]

    assert actions[-1] == "recalibrate"
    assert b.state == "TRANSITION"


def test_blue_bar_is_an_optional_combo_bounce_not_a_loot_target():
    b = Bot()
    b._combo_direction = 1
    blue = BounceSurface(600, 160, 70, 25)

    assert b._blue_bounce_for([blue], intercept_x=400) == blue
    assert b._blue_bounce_for([blue], intercept_x=700) is None


def test_safe_catch_overrides_combo_offset_near_paddle(monkeypatch):
    _seq(monkeypatch, [(400, 900), (400, 950)], (410, 1000))
    monkeypatch.setattr(bot_mod, "detect_special_targets", lambda frame: [])

    b = Bot(deadzone=20)
    b.step(FRAME)

    assert b.step(FRAME) == "hold"
    assert b.last_rebound_mode == "safe_catch"


def test_stalled_low_combo_switches_to_advance_mode(monkeypatch):
    _seq(monkeypatch, [(400, 100)], (400, 1000))
    monkeypatch.setattr(bot_mod, "detect_special_targets", lambda frame: [])

    b = Bot()
    b.returns_without_progress = 4
    b.last_combo = 1
    b.step(FRAME)

    assert b.mission_mode == "advance"


def test_combo_counts_airborne_bounce_then_resets_at_paddle():
    b = Bot()
    b._previous_combo_motion = (12.0, 20.0, 1.0)
    b.combo_bounces = 4

    # Downward -> upward inside the paddle zone ends this combo; it is not
    # counted as an airborne multiplier bounce.
    b._observe_combo((400, 940), (12.0, -20.0, 1.0), paddle_y=1000, height=1080)

    assert b.last_combo == 4
    assert b.best_combo == 4
    assert b.combo_bounces == 0
    assert b._combo_direction == -1


def test_gold_is_only_used_when_it_shares_the_combo_direction():
    b = Bot()
    b._combo_direction = 1  # rebound right

    assert b._use_bonus_rebound(SpecialTarget("gold", 250, 200), 400) is False
    assert b._use_bonus_rebound(SpecialTarget("gold", 600, 200), 400) is True
    assert b._use_bonus_rebound(SpecialTarget("one_up", 250, 200), 400) is True


def test_combo_rebound_offsets_the_cart_without_leaving_catch_range():
    b = Bot()
    b._combo_direction = 1

    aimed = b._aim_for_combo(400, 830)

    assert 30 <= aimed < 400


def test_launch_angle_follows_combo_direction():
    b = Bot()
    assert b.launch_angle() > 0
    b._combo_direction = -1
    assert b.launch_angle() < 0
