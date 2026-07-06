import os

import cv2
import numpy as np

from bootbreaker import capture, config
from tests.conftest import PLAY_REGION, DOCS


def test_calibrate_detects_and_saves(tmp_path):
    main = cv2.imread(os.path.join(DOCS, "main.png"))

    def fake_grabber(sct=None):
        return main

    path = str(tmp_path / "config.json")
    region = capture.calibrate(path, grabber=fake_grabber)

    saved = config.load_config(path)
    assert saved == region
    for key in ("left", "top", "width", "height"):
        assert abs(region[key] - PLAY_REGION[key]) <= 90, (key, region[key])


def test_calibrate_raises_when_no_region(tmp_path):
    def black_grabber(sct=None):
        return np.zeros((100, 100, 3), dtype="uint8")

    path = str(tmp_path / "config.json")
    try:
        capture.calibrate(path, grabber=black_grabber)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
