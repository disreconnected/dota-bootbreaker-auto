import os

import cv2
import numpy as np
import pytest

from bootbreaker import detect
from tests.conftest import DOCS, PLAY_REGION


def _region(name):
    img = cv2.imread(os.path.join(DOCS, name))
    assert img is not None, name
    r = PLAY_REGION
    return img[r["top"]:r["top"] + r["height"], r["left"]:r["left"] + r["width"]]


def _bgr(hsv):
    return tuple(int(v) for v in cv2.cvtColor(
        np.uint8([[hsv]]), cv2.COLOR_HSV2BGR
    )[0, 0])


# Every mid-play frame -> expected ball location (region-local): the centre of
# the brown boot body (not the offset cyan glow), measured from the frames.
BALL_FRAMES = {
    "playing-boot-visible.png": (626, 339),
    "playing-boot-visible-2.png": (396, 576),
    "playing-boot-visible-3.png": (200, 814),
    "playing-boot-visible-4.png": (256, 748),
    "playing-boot-visible-5.png": (122, 1002),
    "playing-boot-visible-6.png": (356, 220),
    "playing-boot-visible-7.png": (177, 848),
    "playing-boot-visible-8.png": (471, 277),
    "playing-boot-visible-with-blue-brick.png": (488, 667),
}

NO_BALL_FRAMES = [
    "main.png",
    "locked-place-before-throw.png",
    "playing-choosing-direction-to-throw.png",
]


def test_detect_special_targets_orders_one_up_before_gold():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(frame, (80, 90), (145, 120), _bgr((65, 230, 230)), -1)
    cv2.rectangle(frame, (280, 180), (390, 205), _bgr((18, 230, 230)), -1)

    targets = detect.detect_special_targets(frame)

    assert [target.kind for target in targets] == ["one_up", "gold"]
    assert abs(targets[0].x - 112) <= 3
    assert abs(targets[1].x - 335) <= 3


def test_detect_indestructible_blue_bar_separately_from_loot_targets():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(frame, (180, 120), (250, 145), _bgr((95, 90, 185)), -1)

    bars = detect.detect_indestructible_bars(frame)

    assert len(bars) == 1
    assert abs(bars[0].x - 215) <= 3
    assert detect.detect_special_targets(frame) == []


def test_breakable_mass_counts_warm_and_pale_blocks_not_blue_bars():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(frame, (80, 100), (140, 130), _bgr((20, 180, 230)), -1)
    cv2.rectangle(frame, (180, 100), (240, 130), _bgr((95, 90, 185)), -1)

    with_gold = detect.estimate_breakable_mass(frame)
    frame[100:131, 80:141] = 0

    assert with_gold > 1_000
    assert detect.estimate_breakable_mass(frame) == 0


def test_detect_paddle_surface_uses_narrow_cyan_rail():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.rectangle(frame, (340, 500), (460, 508), _bgr((95, 220, 220)), -1)

    surface = detect.detect_paddle_surface(frame, cart=(400, 530))

    assert surface is not None
    assert abs(surface.center - 400) <= 4
    assert surface.right - surface.left >= 110


@pytest.mark.parametrize("name,expected", list(BALL_FRAMES.items()))
def test_detect_ball_across_all_playing_frames(name, expected):
    reg = _region(name)
    pos = detect.detect_ball(reg, detect.detect_cart(reg))
    assert pos is not None, f"{name}: ball not found"
    assert abs(pos[0] - expected[0]) <= 30, (name, pos)
    assert abs(pos[1] - expected[1]) <= 30, (name, pos)


@pytest.mark.parametrize("name", NO_BALL_FRAMES)
def test_detect_ball_none_when_not_in_play(name):
    reg = _region(name)
    assert detect.detect_ball(reg, detect.detect_cart(reg)) is None, name


def test_detect_ball_rejects_blue_bricks_specifically():
    # The blue-brick frame has 3 steel-blue bricks (same hue as the glow); the
    # detector must pick the real ball (no brick), verified by its location.
    reg = _region("playing-boot-visible-with-blue-brick.png")
    pos = detect.detect_ball(reg, detect.detect_cart(reg))
    assert pos is not None
    assert 400 <= pos[0] <= 520 and 600 <= pos[1] <= 720, pos  # the ball, not y~296 bricks


def test_detect_ball_rejects_steel_blue_bricks_full_frame():
    # Real full-screen frame with steel-blue bricks beside brown bricks (which
    # satisfy the brown-adjacency test). The dull brick cyan (S~110, V~162) must
    # be rejected by the glow's saturation/value floor, so no false ball.
    img = cv2.imread(os.path.join(DOCS, "full-game-with-ui.png"))
    assert img is not None
    region = detect.detect_play_region(img)
    reg = img[region["top"]:region["top"] + region["height"],
              region["left"]:region["left"] + region["width"]]
    assert detect.detect_ball(reg, detect.detect_cart(reg)) is None


@pytest.mark.parametrize("name", [
    "main.png",                                   # LOCK CART POSITION
    "locked-place-before-throw.png",              # THROW BOOT
    "playing-choosing-direction-to-throw.png",    # THROW BOOT + aim sweep
])
def test_detect_prethrow_true_on_prompt_frames(name):
    reg = _region(name)
    assert detect.detect_prethrow(reg) is True, name


@pytest.mark.parametrize("name", list(BALL_FRAMES))
def test_detect_prethrow_false_mid_play(name):
    reg = _region(name)
    assert detect.detect_prethrow(reg) is False, name


def test_detect_play_region_finds_gold_frame(main_img):
    region = detect.detect_play_region(main_img)
    assert region is not None
    for key in ("left", "top", "width", "height"):
        assert abs(region[key] - PLAY_REGION[key]) <= 90, (key, region[key])


def test_detect_aim_angle_reads_sweep_line(aim_img, crop_region):
    angle = detect.detect_aim_angle(crop_region(aim_img))
    assert angle is not None
    assert 10 <= angle <= 40, angle


def test_detect_aim_angle_none_without_dots(main_img, crop_region):
    assert detect.detect_aim_angle(crop_region(main_img)) is None


def test_aim_fit_exposes_dots_and_direction(aim_img, crop_region):
    # The overlay needs the raw dots + fitted direction, not just the angle.
    angle, pts, unit = detect.aim_fit(crop_region(aim_img))
    assert angle is not None
    assert len(pts) >= 3
    assert unit is not None and len(unit) == 2


def test_detect_cart_found_near_bottom_center(main_img, crop_region):
    pos = detect.detect_cart(crop_region(main_img))
    assert pos is not None
    x, y = pos
    assert 300 <= x <= 560, x
    assert 800 <= y <= 1080, y


def test_detect_cart_centres_on_full_width_not_one_half(main_img, crop_region):
    # The cart's red top is split into two blocks by the central ornament; the
    # centre must be the midpoint of the full span (~418), NOT the centre of the
    # right half alone (~456, the old bug that made it cover only the right side).
    pos = detect.detect_cart(crop_region(main_img))
    assert pos is not None
    assert abs(pos[0] - 418) <= 12, pos
