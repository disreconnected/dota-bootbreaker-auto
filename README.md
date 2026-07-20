# Bootbreaker Autoplayer

A Python bot that auto-plays the Dota 2 **Bootbreaker** arcade minigame — it
launches, aims, and tracks the boot to keep the ball alive as long as possible.

**Windows only.**

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync
```

## Run

```sh
uv run python -m bootbreaker            # play
uv run python -m bootbreaker --debug    # play with live overlay window
```

Open Dota 2 and press **F8** to start/pause the bot (it starts paused).
On first run it auto-detects the play region and saves it to `config.json`.

## Test

```sh
uv run pytest
```

## Credits

This project is based on [jackblk/dota-bootbreaker-auto](https://github.com/jackblk/dota-bootbreaker-auto).

Original implementation by [jackblk](https://github.com/jackblk).
