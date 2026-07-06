"""Screen capture via mss, plus one-time play-region calibration."""

import mss
import numpy as np

from bootbreaker import config, detect


def _to_bgr(shot) -> np.ndarray:
    arr = np.asarray(shot)  # BGRA
    return arr[:, :, :3]  # drop alpha -> BGR (channel order matches OpenCV)


def grab(region: dict, sct=None) -> np.ndarray:
    box = {
        "left": region["left"],
        "top": region["top"],
        "width": region["width"],
        "height": region["height"],
    }
    if sct is not None:
        return _to_bgr(sct.grab(box))
    with mss.mss() as s:
        return _to_bgr(s.grab(box))


def grab_fullscreen(sct=None) -> np.ndarray:
    if sct is not None:
        return _to_bgr(sct.grab(sct.monitors[1]))
    with mss.mss() as s:
        return _to_bgr(s.grab(s.monitors[1]))


def calibrate(config_path: str, grabber=grab_fullscreen) -> dict:
    image = grabber()
    region = detect.detect_play_region(image)
    if region is None:
        raise RuntimeError(
            "Could not find the Bootbreaker play area. Make sure Dota is "
            "visible and windowed, then press F8 again — or edit config.json."
        )
    config.save_config(region, config_path)
    return region
