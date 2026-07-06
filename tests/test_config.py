from bootbreaker import config


def test_load_missing_returns_none(tmp_path):
    assert config.load_config(str(tmp_path / "nope.json")) is None


def test_save_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "config.json")
    region = {"left": 225, "top": 150, "width": 830, "height": 1080}
    config.save_config(region, path)
    assert config.load_config(path) == region
