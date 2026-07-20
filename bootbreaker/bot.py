"""State machine that turns a captured frame into a paddle action.

Launch is driven by the on-screen pre-throw prompt (the space-bar key icon),
not by guessing from ball-absence: we throw only when the game actually shows
"LOCK CART POSITION" / "THROW BOOT". States: AIM (prompt visible, no ball ->
tell main to launch), PLAYING (ball in play -> track), WAIT (neither -> hold).
"""

from __future__ import annotations

from bootbreaker.detect import (
    BounceSurface,
    SpecialTarget,
    detect_ball,
    detect_cart,
    estimate_breakable_mass,
    detect_indestructible_bars,
    detect_paddle_surface,
    detect_prethrow,
    detect_special_targets,
)
from bootbreaker.strategy import (
    adaptive_deadzone,
    decide_move,
    estimate_velocity,
    predict_intercept_x,
)

# How many recent ball sightings to average velocity over. A single-frame
# velocity is far too noisy for the wall-bounce extrapolation (a few px of
# jitter, amplified by the look-ahead, throws the predicted landing hundreds of
# px off and drives the cart the wrong way). Averaging over a couple of frames
# smooths that out.
_VEL_WINDOW = 6
_PREDICTION_CONFIDENCE = 0.55
_GOLD_ATTEMPT_BUDGET = 3
_ONE_UP_ATTEMPT_BUDGET = 5
_CATCH_OVERRIDE_FRAMES = 1.25
_ADVANCE_STALL_RETURNS = 4
_ADVANCE_HARD_STALL_RETURNS = 7
_PRODUCTIVE_COMBO = 3
_TARGET_MATCH_RADIUS = 48
_TARGET_GONE_FRAMES = 3
_AIM_NUDGE = 34
_PADDLE_MARGIN = 30
_COMBO_NUDGE = 46
_COMBO_LAUNCH_ANGLE = 35.0
_BOUNCE_CONFIDENCE = 0.70
_BOUNCE_MIN_SPEED = 3.0
_BOUNCE_DEBOUNCE = 3
_PADDLE_CONTACT_FRAC = 0.16

# The real ball is always moving. A detection pinned to the same spot for
# several frames is a static false positive - a boot-shaped brick in the wall or
# a UI icon that matches the cyan-glow + brown-boot signature. Reject it so the
# cart doesn't sit there oscillating under a phantom after the ball is lost.
_STATIC_EPS = 3  # px; movement at/below this counts as "not moving"
_STATIC_FRAMES = 5  # consecutive still frames -> treat as a static false positive


class Bot:
    def __init__(
        self,
        deadzone: float = 40,
        paddle_frac: float = 0.9,
        chase_gap: int = 6,
    ):
        self.deadzone = deadzone
        self.paddle_frac = paddle_frac
        self.chase_gap = chase_gap
        self.state = "WAIT"
        self._misses = 0
        self._history: list[tuple[int, int]] = []
        self._target_x = None
        self._last_pos = None  # raw last detection, for static-blob rejection
        self._static = 0  # consecutive frames the detection hasn't moved
        self.mission: SpecialTarget | None = None
        self.mission_mode = "survive"
        self._mission_gone = 0
        self._advance_mode = False
        self._advance_empty_frames = 0
        self._ignored_gold: list[SpecialTarget] = []
        self._mission_attempts = 0
        self._force_retarget_gold = False
        self.paddle_returns = 0
        self.returns_without_progress = 0
        self.progress_events = 0
        self._last_return_mass: int | None = None
        self._last_live_board_mass: int | None = None
        self._no_ball_frames = 0
        self._level_transition_candidate = False
        self._transition_recalibrated = False
        # Combo telemetry is inferred from confident velocity reversals. It is
        # independent of OCR/scoreboard layout and therefore survives every
        # level theme and resolution.
        self.combo_bounces = 0
        self.last_combo = 0
        self.best_combo = 0
        self.total_combo_bounces = 0
        self._combo_direction = 1  # +1 = send the next rebound right
        self._previous_combo_motion: tuple[float, float, float] | None = None
        self._bounce_debounce = 0
        # Last detections, exposed for the debug view/logging.
        self.last_ball = None
        self.last_cart = None
        self.last_prethrow = False
        self.last_targets: list[SpecialTarget] = []
        self.last_bounce_surfaces: list[BounceSurface] = []
        self.last_blue_bounce: BounceSurface | None = None
        self.last_paddle_surface = None
        self.last_board_mass = 0
        self.last_velocity: tuple[float, float, float] | None = None
        self.last_intercept: float | None = None
        self.last_deadzone = float(deadzone)
        self.last_rebound_mode = "track"

    def _reset_tracking(self) -> None:
        self._history.clear()
        self._target_x = None
        self.last_velocity = None
        self.last_intercept = None

    def prime_playing(self) -> None:
        """Called right after a throw: reset tracking so the fresh ball is
        picked up cleanly."""
        self.state = "PLAYING"
        self._misses = 0
        self._reset_tracking()
        self._reset_combo()

    def _reset_combo(self) -> None:
        self.combo_bounces = 0
        self.last_combo = 0
        self._previous_combo_motion = None
        self._bounce_debounce = 0

    def launch_angle(self) -> float:
        """Safe diagonal launch angle, alternating sides between attempts."""
        return _COMBO_LAUNCH_ANGLE * self._combo_direction

    def _observe_combo(
        self,
        ball: tuple[int, int],
        motion: tuple[float, float, float] | None,
        paddle_y: float,
        height: int,
        board_mass: int = 0,
    ) -> None:
        """Count a wall/brick bounce from a confident velocity reversal.

        A downward-to-upward reversal in the bottom paddle zone is the end of
        a combo, not a multiplier bounce. Every other reversal is one airborne
        bounce. A short debounce prevents a single collision from being counted
        repeatedly while the rolling velocity window catches up.
        """
        if self._bounce_debounce > 0:
            self._bounce_debounce -= 1
        if motion is None:
            return

        vx, vy, confidence = motion
        previous = self._previous_combo_motion
        self._previous_combo_motion = motion
        if (
            previous is None
            or confidence < _BOUNCE_CONFIDENCE
            or previous[2] < _BOUNCE_CONFIDENCE
            or max(abs(vx), abs(vy), abs(previous[0]), abs(previous[1])) < _BOUNCE_MIN_SPEED
            or self._bounce_debounce > 0
        ):
            return

        horizontal_flip = vx * previous[0] < 0
        vertical_flip = vy * previous[1] < 0
        if not (horizontal_flip or vertical_flip):
            return

        paddle_contact = (
            vertical_flip
            and previous[1] > 0
            and vy < 0
            and ball[1] >= paddle_y - _PADDLE_CONTACT_FRAC * height
        )
        if paddle_contact:
            self.paddle_returns += 1
            self.last_combo = self.combo_bounces
            self.best_combo = max(self.best_combo, self.combo_bounces)
            self.combo_bounces = 0
            if self._last_return_mass is not None:
                progress_threshold = max(300, int(self._last_return_mass * 0.015))
                if board_mass < self._last_return_mass - progress_threshold:
                    self.returns_without_progress = 0
                    self.progress_events += 1
                    if (
                        self.mission is not None
                        and self.mission.kind == "gold"
                        and self.last_rebound_mode == "gold"
                    ):
                        self._force_retarget_gold = True
                else:
                    self.returns_without_progress += 1
            self._last_return_mass = board_mass
            # A mission receives a bounded number of complete paddle-return
            # opportunities. This works across fast and slow layouts, unlike a
            # wall-clock timeout.
            if self.mission is not None:
                self._mission_attempts += 1
            # Alternating sides avoids repeatedly feeding the same cleared lane.
            self._combo_direction *= -1
        else:
            self.combo_bounces += 1
            self.total_combo_bounces += 1
        self._bounce_debounce = _BOUNCE_DEBOUNCE

    def _reset_mission(self) -> None:
        self.mission = None
        self.mission_mode = "survive"
        self._mission_gone = 0
        self._advance_mode = False
        self._advance_empty_frames = 0
        self._ignored_gold.clear()
        self._mission_attempts = 0
        self._force_retarget_gold = False
        self.returns_without_progress = 0
        self._last_return_mass = None

    @staticmethod
    def _same_target(a: SpecialTarget, b: SpecialTarget) -> bool:
        return (
            a.kind == b.kind
            and abs(a.x - b.x) <= _TARGET_MATCH_RADIUS
            and abs(a.y - b.y) <= _TARGET_MATCH_RADIUS
        )

    @staticmethod
    def _choose_target(
        targets: list[SpecialTarget],
        ball: tuple[int, int] | None,
        ignored_gold: list[SpecialTarget] | None = None,
    ) -> SpecialTarget | None:
        ignored_gold = ignored_gold or []
        candidates = [
            target
            for target in targets
            if not (
                target.kind == "gold"
                and any(Bot._same_target(target, ignored) for ignored in ignored_gold)
            )
        ]
        if not candidates:
            return None
        ball_x = ball[0] if ball is not None else 0
        # A 1-UP is always worth more than a gold bar. Within a class, prefer
        # the nearest horizontal target because it needs the smallest paddle
        # deflection and is therefore the safest next bounce.
        priority = {"one_up": 0, "gold": 1}
        return min(candidates, key=lambda t: (priority[t.kind], abs(t.x - ball_x)))

    def _update_mission(
        self, targets: list[SpecialTarget], ball: tuple[int, int] | None
    ) -> SpecialTarget | None:
        """Choose bonuses by return opportunities, not fixed wall-clock time."""
        self.last_targets = targets
        # An abandoned gold becomes eligible again only after it has actually
        # disappeared from the board. This prevents a 15-second retarget from
        # immediately re-selecting the same unreachable block.
        self._ignored_gold = [
            ignored
            for ignored in self._ignored_gold
            if any(self._same_target(ignored, target) for target in targets)
        ]
        # A material board-mass drop immediately after a gold-directed return
        # confirms progress. Retire that target before choosing again; waiting
        # for contour disappearance can waste several frames on its old spot.
        if self._force_retarget_gold:
            if self.mission is not None and self.mission.kind == "gold":
                self._ignored_gold.append(self.mission)
                self.mission = None
                self._mission_attempts = 0
                self._mission_gone = 0
            self._force_retarget_gold = False
        candidate = self._choose_target(targets, ball, self._ignored_gold)

        if self._advance_mode:
            # A newly visible 1-UP is valuable enough to interrupt clearing.
            if candidate is not None and candidate.kind == "one_up":
                self._advance_mode = False
                self.mission = candidate
                self._mission_attempts = 0
                self.mission_mode = "one_up"
                return self.mission
            if targets:
                self._advance_empty_frames = 0
                self.mission_mode = "advance"
                return None
            self._advance_empty_frames += 1
            if self._advance_empty_frames >= _TARGET_GONE_FRAMES:
                self._reset_mission()
            return None

        if self.mission is not None:
            # A newly visible 1-UP outranks an existing gold mission.
            if candidate is not None and candidate.kind == "one_up" and self.mission.kind == "gold":
                self.mission = candidate
                self._mission_gone = 0
                self._mission_attempts = 0
            else:
                observed = next(
                    (target for target in targets if self._same_target(self.mission, target)),
                    None,
                )
                if observed is not None:
                    self.mission = observed
                    self._mission_gone = 0
                else:
                    self._mission_gone += 1
                    gone_limit = (
                        1 if self.mission.kind == "gold" else _TARGET_GONE_FRAMES
                    )
                    if self._mission_gone >= gone_limit:
                        self.mission = None
                        self._mission_gone = 0
                        self._mission_attempts = 0

        if self.mission is None and candidate is not None:
            self.mission = candidate
            self._mission_attempts = 0

        if self.mission is None:
            if self._should_advance():
                self.mission_mode = "advance"
                self._advance_mode = True
            else:
                self.mission_mode = "combo"
            return None

        attempt_budget = (
            _GOLD_ATTEMPT_BUDGET
            if self.mission.kind == "gold"
            else _ONE_UP_ATTEMPT_BUDGET
        )
        if self._mission_attempts >= attempt_budget:
            if self.mission.kind == "gold":
                # Retire that exact gold after enough real return opportunities
                # and immediately choose another target if one exists.
                self._ignored_gold.append(self.mission)
                self.mission = self._choose_target(targets, ball, self._ignored_gold)
                self._mission_gone = 0
                self._mission_attempts = 0
                self.mission_mode = self.mission.kind if self.mission else "combo"
                return self.mission
            self.mission = None
            self.mission_mode = "advance"
            self._advance_mode = True
            return None

        self.mission_mode = self.mission.kind
        return self.mission

    def _should_advance(self) -> bool:
        """Leave bonus hunting only when the board is demonstrably stagnant."""
        if self.returns_without_progress >= _ADVANCE_HARD_STALL_RETURNS:
            return True
        return (
            self.returns_without_progress >= _ADVANCE_STALL_RETURNS
            and self.last_combo < _PRODUCTIVE_COMBO
        )

    @staticmethod
    def _aim_for_target(
        catch_center: float,
        target: SpecialTarget,
        width: int,
        max_offset: float = _AIM_NUDGE,
        landing_x: float | None = None,
    ) -> float:
        """Offset the cart slightly opposite the desired outgoing direction.

        Moving the cart centre left makes the incoming boot hit its right side,
        nudging the rebound right (and conversely for left).  The offset is
        capped well inside the cart's catch width, so mission steering never
        replaces the primary job of keeping the boot alive.
        """
        landing_x = catch_center if landing_x is None else landing_x
        delta = target.x - landing_x
        if abs(delta) < 8:
            return catch_center
        nudge = min(_AIM_NUDGE, max_offset, max(12.0, abs(delta) * 0.12))
        aimed = catch_center - nudge if delta > 0 else catch_center + nudge
        return max(_PADDLE_MARGIN, min(width - _PADDLE_MARGIN, aimed))

    def _aim_for_combo(
        self, catch_center: float, width: int, max_offset: float = _COMBO_NUDGE
    ) -> float:
        """Create a wide but still catch-safe diagonal rebound.

        Low-scoring combos receive a slightly stronger edge hit to search for a
        better wall/brick path. Once the bot is already finding bounces, the
        offset relaxes a little to protect the run.
        """
        nudge = _COMBO_NUDGE if self.last_combo < 2 else _COMBO_NUDGE * 0.82
        nudge = min(nudge, max_offset)
        aimed = catch_center - self._combo_direction * nudge
        return max(_PADDLE_MARGIN, min(width - _PADDLE_MARGIN, aimed))

    def _use_bonus_rebound(self, mission: SpecialTarget, intercept_x: float) -> bool:
        """1-UPs always override; gold must share the combo's outgoing side."""
        if mission.kind == "one_up":
            return True
        delta = mission.x - intercept_x
        return abs(delta) < 24 or (delta > 0) == (self._combo_direction > 0)

    def _blue_bounce_for(
        self, bars: list[BounceSurface], intercept_x: float
    ) -> BounceSurface | None:
        """Pick an indestructible bar on the current combo side as a rebound.

        Blue bars are never score missions. They are only exposed here when no
        bonus can share the current outgoing route, where their predictable
        reflection can add one more airborne bounce.
        """
        candidates = [
            bar
            for bar in bars
            if (bar.x - intercept_x) * self._combo_direction >= 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda bar: abs(bar.x - intercept_x) + bar.y * 0.15)

    def step(self, frame) -> str:
        h, w = frame.shape[:2]
        cart = detect_cart(frame)
        self.last_cart = cart
        surface = detect_paddle_surface(frame, cart)
        self.last_paddle_surface = surface
        ball = detect_ball(frame, cart)
        self.last_ball = ball
        targets = detect_special_targets(frame)
        bars = detect_indestructible_bars(frame)
        self.last_board_mass = estimate_breakable_mass(frame)
        self.last_bounce_surfaces = bars
        self.last_blue_bounce = None

        if ball is not None:
            self._last_live_board_mass = self.last_board_mass
            self._no_ball_frames = 0
            self._level_transition_candidate = False
            self._transition_recalibrated = False
        else:
            self._no_ball_frames += 1
            if self._last_live_board_mass:
                change = abs(self.last_board_mass - self._last_live_board_mass)
                if change / self._last_live_board_mass >= 0.22:
                    self._level_transition_candidate = True

        # --- reject static false positives ---
        # A detection that hasn't moved for a few frames isn't the ball (the
        # ball always moves); it's a fixed brick/icon. Drop it.
        if (
            ball is not None
            and self._last_pos is not None
            and abs(ball[0] - self._last_pos[0]) <= _STATIC_EPS
            and abs(ball[1] - self._last_pos[1]) <= _STATIC_EPS
        ):
            self._static += 1
        else:
            self._static = 0
        self._last_pos = ball
        if ball is not None and self._static >= _STATIC_FRAMES:
            ball = None
            self._target_x = None

        prethrow = detect_prethrow(frame)
        if (
            not prethrow
            and self._level_transition_candidate
            and self._no_ball_frames >= 8
        ):
            prethrow = detect_prethrow(frame, threshold=0.68)
        self.last_prethrow = prethrow

        # Targets are static enough to identify while the boot travels. The
        # chosen mission persists across frames, avoiding aim flip-flops from a
        # one-pixel contour shift.
        mission = self._update_mission(targets, ball)

        # --- PLAYING: a ball is in play, track it ---
        if ball is not None:
            self.state = "PLAYING"
            self._misses = 0
            self._history.append(ball)
            if len(self._history) > _VEL_WINDOW:
                self._history.pop(0)
            # Follow the boot's x for pre-positioning.  On a confident descent,
            # predict the complete wall-bounced landing independent of speed;
            # median velocity prevents one noisy capture from dominating it.
            motion = estimate_velocity(self._history)
            self.last_velocity = motion
            self.last_intercept = None
            frames_to_paddle = None
            self._target_x = ball[0]
            if motion is not None:
                vx, vy, confidence = motion
                paddle_y = (
                    surface.y
                    if surface is not None
                    else (cart[1] if cart is not None else int(self.paddle_frac * h))
                )
                self._observe_combo(ball, motion, paddle_y, h, self.last_board_mass)
                if vy > 0:
                    frames_to_paddle = (paddle_y - ball[1]) / vy
                    if frames_to_paddle >= 0 and confidence >= _PREDICTION_CONFIDENCE:
                        intercept = predict_intercept_x(
                            ball[0], ball[1], vx, vy, paddle_y, w
                        )
                        self.last_intercept = intercept
                        rail_shift = (
                            surface.center - cart[0]
                            if surface is not None and cart is not None
                            else 0
                        )
                        catch_center = intercept - rail_shift
                        max_offset = (
                            max(0.0, min(_COMBO_NUDGE, surface.half_width - 18.0))
                            if surface is not None
                            else 0.0
                        )
                        if frames_to_paddle <= _CATCH_OVERRIDE_FRAMES:
                            # At the last moment, landing certainty outweighs
                            # every multiplier, gold bar, and rebound surface.
                            self._target_x = catch_center
                            self.last_rebound_mode = "safe_catch"
                        else:
                            use_bonus = mission is not None and self._use_bonus_rebound(
                                mission, intercept
                            )
                            self._target_x = self._aim_for_combo(
                                catch_center, w, max_offset
                            )
                            self.last_rebound_mode = "combo"
                            if use_bonus:
                                self._target_x = self._aim_for_target(
                                    catch_center,
                                    mission,
                                    w,
                                    max_offset,
                                    landing_x=intercept,
                                )
                                self.last_rebound_mode = mission.kind
                            else:
                                blue_bounce = self._blue_bounce_for(bars, intercept)
                                if blue_bounce is not None:
                                    self.last_blue_bounce = blue_bounce
                                    self.last_rebound_mode = "blue_bounce"
            self.last_deadzone = adaptive_deadzone(self.deadzone, frames_to_paddle)
            if cart is None or self._target_x is None:
                return "hold"
            return decide_move(self._target_x, cart[0], self.last_deadzone) or "hold"

        # --- no ball ---
        self._misses += 1
        self._history.clear()
        if (
            self._level_transition_candidate
            and self._no_ball_frames >= 24
            and not self._transition_recalibrated
        ):
            self.state = "TRANSITION"
            self._transition_recalibrated = True
            return "recalibrate"
        if prethrow:
            # The game is waiting for us to throw. This is the ONLY launch
            # trigger, so we never fire at a random moment.
            self.state = "AIM"
            self._reset_tracking()
            # A board with no remaining special target is normally the level
            # transition. It is safe to arm bonus missions again next level.
            if not targets:
                self._reset_mission()
            return "launch"

        # No ball and no prompt: a brief transition (the ball just left the cart
        # / a detection blip). Keep steering to the last target through the
        # blind-spot gap, then hold until the ball reappears or the prompt shows.
        self.state = "WAIT"
        if (
            self._target_x is not None
            and cart is not None
            and self._misses <= self.chase_gap
        ):
            return decide_move(self._target_x, cart[0], self.deadzone) or "hold"
        self._target_x = None
        return "hold"
