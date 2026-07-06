from bootbreaker.main import build_parser


def test_flags_default_false():
    args = build_parser().parse_args([])
    assert args.debug is False
    assert args.recalibrate is False


def test_flags_can_be_set():
    args = build_parser().parse_args(["--debug", "--recalibrate"])
    assert args.debug is True
    assert args.recalibrate is True
