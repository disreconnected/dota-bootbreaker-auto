import os

import cv2
import pytest

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")
PLAY_REGION = {"left": 225, "top": 150, "width": 830, "height": 1080}


def _load(name):
    img = cv2.imread(os.path.join(DOCS, name))
    assert img is not None, f"could not read {name}"
    return img


@pytest.fixture
def main_img():
    return _load("main.png")


@pytest.fixture
def playing_img():
    return _load("playing-boot-visible.png")


@pytest.fixture
def blue_brick_img():
    # Level 2 frame with pale steel-blue bricks near the top — must not fool
    # the cyan-glow ball detector.
    return _load("playing-boot-visible-with-blue-brick.png")


@pytest.fixture
def playing_img2():
    # A harder mid-play frame where the boot's cyan glow is only a small crescent.
    return _load("playing-boot-visible-2.png")


@pytest.fixture
def playing_img3():
    # Another hard mid-play frame: boot descending low and to the left.
    return _load("playing-boot-visible-3.png")


@pytest.fixture
def aim_img():
    # The THROW BOOT screen showing the sweeping launch-direction dotted line.
    return _load("playing-choosing-direction-to-throw.png")


@pytest.fixture
def crop_region():
    def _crop(img):
        r = PLAY_REGION
        return img[r["top"]:r["top"] + r["height"], r["left"]:r["left"] + r["width"]]

    return _crop
