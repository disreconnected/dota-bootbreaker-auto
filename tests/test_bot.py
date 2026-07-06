import numpy as np

import bootbreaker.bot as bot_mod
from bootbreaker.bot import Bot

FRAME = np.zeros((1080, 830, 3), dtype="uint8")  # dummy; detect is patched


def _seq(monkeypatch, balls, cart, prethrow=False):
    it = iter(balls)
    monkeypatch.setattr(bot_mod, "detect_ball", lambda frame, cart=None: next(it))
    monkeypatch.setattr(bot_mod, "detect_cart", lambda frame: cart)
    monkeypatch.setattr(bot_mod, "detect_prethrow", lambda frame: prethrow)


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


def test_holds_within_deadzone(monkeypatch):
    _seq(monkeypatch, [(410, 100), (410, 300)], (400, 1000))
    b = Bot(deadzone=20)
    b.step(FRAME)
    assert b.step(FRAME) == "hold"


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
