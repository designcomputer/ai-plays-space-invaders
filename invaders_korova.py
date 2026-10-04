"""Jev-style Space Invaders agent for the korovatron.co.uk canvas game.

Unlike the elgoog DOM game, this game exposes its full state as JavaScript
globals (cannon, fleet, shields, missiles, ufo, score, lives, gameState), so
we read state directly from JS — no canvas pixel analysis.

The four destructible bunkers (shields) sit between the cannon and the
invaders and block upward shots — the same "obstacle" problem as elgoog's
Google logo. We compute per-column reachability (is there a clear vertical
path through the bunker band?) and feed it to the decision model so it prefers
targets it can actually hit, instead of wasting shots on the bunkers.

Architecture (constrained decision, not open-ended generation):
  observe  -> read JS globals, compute bunker reachability
  decide   -> System One model picks ONE offered target (invader column / UFO)
  act      -> deterministic: move cannon to align x, reflex-fire when clear

Controls: ArrowLeft/ArrowRight move the cannon, Space fires.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass

from playwright.sync_api import Page, sync_playwright

from systemone import SystemOneError, systemone

URL = "https://www.korovatron.co.uk/spaceinvaders/"
DEFAULT_BASE_URL = os.environ.get("SYSTEMONE_BASE_URL", "http://sriai:11434")

# Canvas geometry (baseWidth=896, baseHeight=1024)
BASE_W = 896
BASE_H = 1024
CANNON_Y = 864
SHIELD_Y = 768
SHIELD_ROWS = 16
SHIELD_COLS = 24
TILE_PX = 4  # pixelSize = tileSize * 4
SHIELD_H = SHIELD_ROWS * TILE_PX  # 64px
MISSILE_SPEED = 750.0  # player bullet speed, px/s
# Heuristic buffer before the game's instant-loss "invaders reach the bottom"
# line (we don't have the exact threshold from source, since it's not
# included in this standalone copy) — prioritize a column once its lowest
# invader crosses this, overriding score/descent strategy entirely. y
# increases downward (toward the cannon at CANNON_Y), so this must be well
# above SHIELD_Y's far side (832) to leave real reaction time.
INVADER_DANGER_Y = 800.0


def norm(x: float) -> int:
    """Canvas x (0..896) -> 0..100 for the model."""
    return max(0, min(100, round(x / BASE_W * 100)))


def record_to_mp4(webm_path: str, mp4_path: str) -> None:
    """Convert a Playwright-recorded .webm to .mp4 via ffmpeg (must be on
    PATH). Playwright only records video as .webm; this is the one extra
    step to get an .mp4 out of it."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_path],
        check=True, capture_output=True,
    )


# Read the whole game state from JS globals in one call.
_STATE_JS = r"""() => {
  const o = {gameState, score, lives, shieldsOn};
  o.cannonX = cannon ? cannon.x : null;
  o.fleet = (fleet||[]).filter(i=>!i.isDead()).map(i=>({x:i.x, y:i.y, t:i.type}));
  o.missile = (typeof activeMissile!=='undefined' && activeMissile && activeMissile.active)
              ? {x:activeMissile.x, y:activeMissile.y} : null;
  o.invMissiles = (invaderMissiles||[]).filter(m=>m.active).map(m=>({x:m.x, y:m.y}));
  o.ufo = (ufo && ufo.isActive()) ? {x:ufo.x, y:ufo.y, speed:ufo.speed, width:ufo.width} : null;
  // bunker tiles: per shield, per column, is the column fully open (all tiles destroyed)?
  o.shields = (shields||[]).map(s => {
    const cols = [];
    for (let c=0;c<s.cols;c++){
      let open = true;
      for (let r=0;r<s.rows;r++){ if (s.tiles[r][c] < 3){ open=false; break; } }
      cols.push(open);
    }
    return {x:s.x, cols};
  });
  return o;
}"""


@dataclass
class PlayResult:
    seconds: float
    score: str = ""
    lives: int = 0
    ticks: int = 0
    model_calls: int = 0
    error: str = ""
    shots: int = 0
    hits: int = 0


class KorovaAgent:
    def __init__(self, *, headless: bool = True, model: str = "clef:27b",
                 base_url: str | None = None, policy: str = "model",
                 keep_alive: str = "30m", aim_align: int = 4,
                 aggression: float = 0.5,
                 dodge_trigger_y: float = 600.0, dodge_radius: float = 56.0,
                 record_video_dir: str | None = None):
        # clef:27b beats tev1:4b here (higher score + better dodging/survival)
        # at ~2.6x latency (240ms vs 92ms/call) — still fast enough for the game.
        self.headless = headless
        self.model = model
        self.base_url = base_url or DEFAULT_BASE_URL
        self.policy = policy  # "model" (System One) or "baseline" (deterministic)
        self.keep_alive = keep_alive
        self.aim_align = aim_align  # norm-x tolerance for "aligned"
        # 0.0 = pure survival (always slow the descent, stick to covered outer
        # columns), 1.0 = pure score (chase the deepest/richest column even if
        # farther away or exposed). Scaled down by remaining lives at runtime
        # (see _decide_baseline) so an aggressive setting still plays safe on
        # the last life.
        self.aggression = max(0.0, min(1.0, aggression))
        # Reactive-dodge tuning: the shield band bottom (832) sits only ~32px
        # above the cannon (864), so there's little room to react once a
        # missile clears the bunkers. Trigger the dodge well before that.
        self.dodge_trigger_y = dodge_trigger_y
        self.dodge_radius = dodge_radius
        # When set, records the session to a .webm file in this directory
        # (Playwright's native format — see record_to_mp4() to convert).
        self.record_video_dir = record_video_dir
        self.video_path: str | None = None  # set in __exit__ once finalized
        self._current_target: str | None = None
        self._current_target_x: float | None = None  # for drift-tolerant persistence
        self._start_lives: int | None = None
        self._pw = None
        self._browser = None
        self._context = None
        self._page: Page | None = None

    # ---------- lifecycle ----------
    def __enter__(self) -> "KorovaAgent":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        viewport = {"width": 1280, "height": 900}
        if self.record_video_dir:
            self._context = self._browser.new_context(
                viewport=viewport,
                record_video_dir=self.record_video_dir,
                record_video_size=viewport,
            )
            self._page = self._context.new_page()
        else:
            self._page = self._browser.new_page(viewport=viewport)
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._page and self.record_video_dir:
                video = self._page.video
                self._page.close()  # video only finalizes once its page closes
                if video:
                    self.video_path = video.path()
        finally:
            try:
                if self._context:
                    self._context.close()
            finally:
                try:
                    if self._browser:
                        self._browser.close()
                finally:
                    if self._pw:
                        self._pw.stop()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("agent not started")
        return self._page

    # ---------- setup ----------
    def _start_game(self) -> None:
        page = self.page
        page.goto(URL, timeout=45000, wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        # Focus the page, then start. Space should start it; if a focus/timer
        # quirk blocks that, call the game's own newGame() (equivalent to the
        # start button) so we reliably reach gameplay.
        page.evaluate("() => window.focus()")
        page.mouse.click(BASE_W / 2, 300)
        page.wait_for_timeout(400)
        page.keyboard.press("Space")
        for _ in range(30):
            st = page.evaluate(_STATE_JS)
            if st.get("gameState") == 1:
                return
            page.wait_for_timeout(300)
        # fallback: start directly
        page.evaluate("() => { try { newGame(); } catch(e){} }")
        for _ in range(30):
            st = page.evaluate(_STATE_JS)
            if st.get("gameState") == 1:
                return
            page.wait_for_timeout(300)
        raise RuntimeError("game did not start (gameState != 1)")

    # ---------- observe ----------
    def _state(self) -> dict:
        return self.page.evaluate(_STATE_JS)

    def _reachable_x(self, st: dict, x: float) -> bool:
        """Is there a clear vertical path at canvas-x `x` through the bunker band?
        True if no intact bunker tile sits at that x."""
        for s in st.get("shields", []):
            x0 = s["x"]
            x1 = x0 + SHIELD_COLS * TILE_PX
            if x0 - 2 <= x < x1:
                c = int((x - x0) // TILE_PX)
                if 0 <= c < SHIELD_COLS:
                    return s["cols"][c]  # open only if that column is fully destroyed
        return True  # not over any bunker -> clear

    def _covered_x(self, st: dict, x: float) -> bool:
        """Is the cannon protected by a bunker when aligned with column x?
        The cannon centers on x+16 when aligned; it's covered if that point is
        under a bunker (bunkers block enemy fire coming down)."""
        cx = x + 16  # cannon center when aligned with this column
        for s in st.get("shields", []):
            x0 = s["x"]
            x1 = x0 + SHIELD_COLS * TILE_PX
            if x0 - 8 <= cx < x1 + 8:
                return True
        return False

    def _threat(self, st: dict, cannon_x: float) -> dict | None:
        """Most urgent incoming enemy missile on a collision course with the
        cannon's CURRENT x (not the aim target). Checked every tick inside
        _aim_at so a missile that appears mid-chunk is dodged immediately,
        not only when the next decision chunk starts."""
        cannon_cx = cannon_x + 32
        worst = None
        for m in st.get("invMissiles", []):
            if m["y"] < self.dodge_trigger_y:
                continue
            if abs(m["x"] - cannon_cx) > self.dodge_radius:
                continue
            if worst is None or m["y"] > worst["y"]:  # lowest = most urgent
                worst = m
        return worst

    def _column_has_invader(self, st: dict, x: float) -> bool:
        """Is there a live invader anywhere in the 32px-wide column at x? The
        UFO flies above the whole fleet, so a shot aimed at it is intercepted
        by any surviving invader in that column long before it gets that
        high — this is the dominant reason UFO shots were missing even once
        aim/lead/bunker-reachability were all correct."""
        return any(inv["x"] <= x < inv["x"] + 32 for inv in st.get("fleet", []))

    def _ufo_reachable_x(self, st: dict, x: float) -> bool:
        """Can a bullet aimed at x actually reach the UFO's altitude? Needs
        both a clear bunker column AND no surviving invader in the way."""
        return self._reachable_x(st, x) and not self._column_has_invader(st, x)

    def _targets(self, st: dict) -> dict[str, dict]:
        """Offered target set: lowest-row invader columns + UFO, each annotated
        with reachability (can a bullet fired at its x get through the bunkers?)."""
        targets: dict[str, dict] = {}
        # Group invaders into columns by x (they move together). For each
        # column, the lowest invader is what a bullet hits first — clear the
        # column bottom-to-top for max points (10->20->20->30).
        cols: dict[float, list[dict]] = {}
        for inv in st.get("fleet", []):
            cols.setdefault(round(inv["x"]), []).append(inv)
        for x in sorted(cols):
            members = cols[x]
            lowest = max(members, key=lambda m: m["y"])  # closest to player
            nx = norm(x)
            reach = self._reachable_x(st, x)
            cover = self._covered_x(st, x)
            depth = len(members)
            tags = []
            tags.append('open' if reach else 'BLOCKED')
            if cover:
                tags.append('cover')
            if lowest["y"] >= SHIELD_Y:
                tags.append('LOW')
            targets[f"c{nx}"] = {"x": nx, "cx": x, "reachable": reach,
                                 "cover": cover, "depth": depth, "lowest_y": lowest["y"],
                                 "label": f"col x={nx} {depth} deep, lowest_y={round(lowest['y'])} "
                                          f"({'/'.join(tags)})"}
        if st.get("ufo"):
            nx = norm(st["ufo"]["x"])
            targets["ufo"] = {"x": nx, "cx": st["ufo"]["x"], "depth": 0,
                              "reachable": self._ufo_reachable_x(st, st["ufo"]["x"]),
                              "label": f"UFO (50-300pts) x={nx}"}
        return targets

    def _compact_state(self, st: dict, targets: dict[str, dict]) -> dict:
        cx = norm(st.get("cannonX") or (BASE_W / 2))
        # nearest incoming enemy missile to the cannon
        threat = None
        if st.get("invMissiles"):
            px = st.get("cannonX") or BASE_W / 2
            threat = min(st["invMissiles"], key=lambda m: abs(m["x"] - px))
        reachable = [v["x"] for v in targets.values() if v.get("reachable")]
        blocked = [v["x"] for v in targets.values() if not v.get("reachable")]
        lives = st.get("lives")
        start_lives = self._start_lives or max(lives or 1, 1)
        life_frac = max(0.0, min(1.0, (lives or 0) / start_lives)) if start_lives else 1.0
        return {
            "cannon_x": cx,
            "target_list": [f"{k}={v['x']}" for k, v in targets.items()],
            "reachable_columns": reachable or None,
            "blocked_columns": blocked or None,
            "nearest_enemy_missile_x": norm(threat["x"]) if threat else None,
            "missile_below_cannon": bool(threat and abs(threat["x"] - (st.get("cannonX") or 0)) <= 12),
            "ufo_out": st.get("ufo") is not None,
            "player_missile_active": st.get("missile") is not None,
            "score": st.get("score"),
            "lives": lives,
            "life_frac": round(life_frac, 2),
            "aggression": self.aggression,
        }

    # ---------- decide ----------
    def _decide(self, cs: dict, targets: dict[str, dict]) -> str | None:
        if not targets:
            return None
        if len(targets) == 1:  # choice needs 2-26 candidates
            return next(iter(targets))
        state = {
            "cannon_x": cs["cannon_x"],
            "nearest_enemy_missile_x": cs["nearest_enemy_missile_x"],
            "missile_below_cannon": cs["missile_below_cannon"],
            "ufo_out": cs["ufo_out"],
            "player_missile_active": cs["player_missile_active"],
            "score": cs["score"],
            "lives": cs["lives"],
            "life_frac": cs["life_frac"],
            "aggression": cs["aggression"],
        }
        questions = {
            "target": {
                "type": "choice",
                "instructions": (
                    "You are playing Space Invaders. The cannon fires "
                    "automatically; you choose which COLUMN to focus on. All x "
                    "are 0-100 (0=left, 100=right). KEY STRATEGY — SLOW THE "
                    "DESCENT: the alien fleet drops a row each time it bounces "
                    "off a screen edge, and it bounces off its OUTERMOST "
                    "surviving column. While many columns remain, clear the "
                    "OUTER columns (near x=0 or x=100) first to shrink the "
                    "fleet and slow its descent — this buys survival time. "
                    "Columns tagged 'cover' have a bunker beneath them (safe "
                    "from enemy fire); PREFER covered outer columns, since the "
                    "bare edges are exposed. BOTTOM ROW: each column's label "
                    "shows lowest_y, how close its nearest invader is to the "
                    "bottom of the screen (higher = closer = more dangerous); "
                    "columns tagged 'LOW' are nearing the bottom and should be "
                    "prioritized over value/distance/descent-strategy, since "
                    "an invader reaching the bottom is instant game over. "
                    "Once few columns remain (and none are LOW), weigh value "
                    "vs. distance: each column's label shows how many "
                    "invaders deep it is (more deep = more remaining points, "
                    "cleared bottom-to-top). Bunkers BLOCK most shots: "
                    "reachable_columns get through, blocked_columns waste the "
                    "bullet — STRONGLY prefer reachable. The UFO is worth "
                    "50-300pts, pick it when ufo_out=true and reachable. If "
                    "missile_below_cannon=true, pick a column that also moves "
                    "you away from nearest_enemy_missile_x. RISK TOLERANCE: "
                    "aggression (0=play safe, 1=maximize score) combined with "
                    "life_frac (1.0=full lives, lower=fewer lives left) sets "
                    "how much risk to take — when aggression*life_frac is "
                    "high, favor a deeper/richer column even if it's farther "
                    "or uncovered; when it's low, stick to the nearest covered "
                    "outer column and the descent-slowing strategy. Choose one "
                    "column."
                ),
                "criteria": {k: v["label"] for k, v in targets.items()},
            },
        }
        resp = systemone(self.model, state=state, questions=questions,
                         keep_alive=self.keep_alive, base_url=self.base_url)
        return resp.answers["target"].choice

    def _decide_baseline(self, cs: dict, targets: dict[str, dict]) -> str | None:
        """Deterministic focus-column picker. UFO handled in the main loop.

        SLOW THE DESCENT: the fleet bounces (and drops a row) when its outermost
        invader hits a screen edge. While the fleet is wide, clear the OUTER
        columns first — that moves the bounce point inward, so the fleet takes
        longer to reach the edges and descends more slowly (more survival time).

        AGGRESSION: self.aggression (0=survival, 1=score) is scaled down by
        life_frac (lives remaining / starting lives), so an aggressive setting
        automatically plays it safe once the player is down to its last life.
        Effective aggression both raises the "fleet is wide" threshold (an
        aggressive agent abandons descent-slowing sooner) and, once in
        focus-fire mode, weighs a column's remaining VALUE (depth = invaders
        left = points left) against the travel cost to reach it, instead of
        always taking the nearest column."""
        if not targets:
            return None
        cx = cs["cannon_x"]
        cols = {k: v for k, v in targets.items() if k != "ufo"}
        reachable = {k: v for k, v in cols.items() if v.get("reachable")}
        pool = reachable or cols
        if not pool:
            return None

        lives = cs.get("lives") or 0
        start_lives = self._start_lives or max(lives, 1)
        life_frac = max(0.0, min(1.0, lives / start_lives)) if start_lives else 1.0
        eff_aggression = self.aggression * life_frac

        wide_threshold = 8 + round(eff_aggression * 8)  # 8 (survival) .. 16 (aggressive)
        if len(cols) >= wide_threshold:
            # Prioritize OUTER columns, but only ones with bunker COVER (the bare
            # edges are exposed to enemy fire). Clearing a covered outer column
            # shrinks the fleet (slower descent) while keeping the cannon safe.
            xs = sorted(pool, key=lambda k: pool[k]["x"])
            outer = set(xs[:3] + xs[-3:])  # leftmost 3 + rightmost 3
            covered_outer = {k for k in outer if pool[k].get("cover")}
            if covered_outer:
                return min(covered_outer, key=lambda k: abs(pool[k]["x"] - cx))
            # no covered outer column: fall back to nearest outer (still slows
            # descent) rather than giving up the strategy entirely
            if outer:
                return min(outer, key=lambda k: abs(pool[k]["x"] - cx))

        # Narrow fleet (or aggressive enough to skip descent-slowing): score by
        # remaining value vs. travel cost. Survival-leaning settings stay
        # nearest-column (as before); score-leaning settings reach further for
        # a deeper, richer column. URGENCY (how close the column's lowest
        # invader is to the danger line) is weighted heavily so a column
        # nearing the bottom gets prioritized well before the hard override
        # in play() would kick in — not just "clear the bottom row" once it's
        # almost too late.
        def score(k: str) -> float:
            v = pool[k]
            value = v.get("depth", 0) / 4.0  # normalize (max column depth ~4-5)
            dist_cost = abs(v["x"] - cx) / 100.0
            cover_bonus = 0.1 if v.get("cover") else 0.0
            urgency = max(0.0, (v.get("lowest_y", 0) - SHIELD_Y) / (INVADER_DANGER_Y - SHIELD_Y))
            return (eff_aggression * value - (1.0 - 0.5 * eff_aggression) * dist_cost
                    + cover_bonus + 1.5 * urgency)

        return max(pool, key=score)

    # ---------- act ----------
    def _reflex_fire(self, st: dict) -> bool:
        if st.get("gameState") == 1 and not st.get("missile"):
            self.page.keyboard.press("Space")
            return True
        return False

    def _aim_at(self, target_key: str, duration: float, force_fire: bool = False) -> int:
        """Move cannon toward the live x of `target_key`, firing throughout.
        Re-reads the target's current x each iteration (the fleet drifts).

        force_fire=True skips the reachability gate on firing (still requires
        alignment) — used for the bottom-row emergency override in play(): if
        a dangerously-low invader's only path is through a bunker, shooting
        the bunker to erode it is strictly better than holding fire.

        DRIFT-TOLERANT KEY RESOLUTION: a column's key (e.g. "c42") is derived
        from its rounded normalized x, so drift across a rounding boundary
        mid-chunk renames the key — a plain dict lookup would then see
        "target gone" and bail out, right when the fleet is drifting fastest
        (e.g. the moment it reverses direction off a screen edge). Instead we
        track the target by physical x and re-resolve to the nearest column
        if the exact key disappears.

        LEADING THE TARGET: the UFO moves at roughly 2x the cannon's speed,
        and even the fleet's slower drift adds up over a bullet's ~1s flight
        to a tall column's lowest invader — especially right as the fleet
        reverses direction mid-flight — so a shot aimed at the target's
        CURRENT x can miss. We lead both: the UFO exposes its exact velocity
        as `ufo.speed` (used directly — numerically differencing sampled
        positions was tried first and was too noisy, since per-tick position
        deltas are tiny relative to page.evaluate() round-trip jitter); the
        fleet doesn't expose a velocity field and moves in discrete steps
        rather than continuously, so we estimate it from a short rolling
        window of samples (noisy single-tick deltas average out).

        Returns the number of real missiles launched."""
        fired = 0
        had_missile = False
        key = None
        last_cx: float | None = None  # physical x of the target, across key renames
        fleet_hist: deque[tuple[float, float]] = deque(maxlen=12)  # (t, x), ~240ms window
        end = time.time() + duration
        while time.time() < end:
            st = self._state()
            if st.get("gameState") != 1:
                break
            cx = st.get("cannonX")
            if cx is None:
                break

            threat = self._threat(st, cx)
            if threat is not None:
                # DODGE: an enemy missile is on a collision course with the
                # cannon's current position. Break off aiming and move away
                # immediately — don't wait for this chunk to end.
                dodge_key = "ArrowRight" if threat["x"] < cx + 32 else "ArrowLeft"
                if dodge_key != key:
                    if key:
                        self.page.keyboard.up(key)
                    self.page.keyboard.down(dodge_key)
                    key = dodge_key
                if st.get("missile") and not had_missile:
                    fired += 1
                had_missile = st.get("missile") is not None
                time.sleep(0.02)
                continue

            # live target (it may have drifted since we last looked)
            all_targets = self._targets(st)
            tgt = all_targets.get(target_key)
            if tgt is None and target_key != "ufo" and last_cx is not None:
                _, near_v = min(
                    ((k, v) for k, v in all_targets.items() if k != "ufo"),
                    key=lambda kv: abs(kv[1]["cx"] - last_cx), default=(None, None))
                if near_v is not None and abs(near_v["cx"] - last_cx) <= 16:
                    tgt = near_v
            if tgt is None:
                break  # target really gone (cleared, or drifted too far to be the same column)
            last_cx = tgt["cx"]

            aim_cx = tgt["cx"]  # left-edge x to aim at (lead-adjusted below)
            half_w = 16.0  # invaders are 32px wide
            reachable = tgt.get("reachable", True)
            ufo = st.get("ufo")
            if target_key == "ufo" and ufo:
                ux, uy = ufo["x"], ufo["y"]
                half_w = ufo.get("width", 16) / 2.0
                lead_time = max(0.0, CANNON_Y - uy) / MISSILE_SPEED
                aim_cx = ux + ufo.get("speed", 0.0) * lead_time
                # reachability must be checked at the PREDICTED x: that's
                # where the bullet will actually cross the bunker band and
                # fleet (a surviving invader there intercepts it too).
                reachable = self._ufo_reachable_x(st, aim_cx)
            else:
                now = time.time()
                fleet_hist.append((now, tgt["cx"]))
                t0, x0 = fleet_hist[0]
                vel = (tgt["cx"] - x0) / (now - t0) if now - t0 > 0.05 else 0.0
                lead_time = max(0.0, CANNON_Y - tgt.get("lowest_y", CANNON_Y)) / MISSILE_SPEED
                aim_cx = tgt["cx"] + vel * lead_time
                reachable = self._reachable_x(st, aim_cx)

            # The missile fires from cannon.x+32 (cannon center). Align the
            # cannon center to the target's center: cannon.x = target_left - half_w.
            aim_x = aim_cx - half_w
            diff = aim_x - cx
            aligned = abs(diff) <= self.aim_align * TILE_PX
            if not aligned:
                k = "ArrowRight" if diff > 0 else "ArrowLeft"
                if k != key:
                    if key:
                        self.page.keyboard.up(key)
                    self.page.keyboard.down(k)
                    key = k
            elif key:
                self.page.keyboard.up(key)
                key = None
            # PRECISION: only fire when aligned AND the live column is
            # reachable. One bullet at a time, and a misaligned/blocked shot
            # takes ~1s to clear the top or just hits the bunker — don't
            # waste it.
            if aligned and not st.get("missile") and (reachable or force_fire):
                self.page.keyboard.press("Space")
            if st.get("missile") and not had_missile:
                fired += 1
            had_missile = st.get("missile") is not None
            time.sleep(0.02)
        if key:
            self.page.keyboard.up(key)
        return fired

    # ---------- main loop ----------
    def play(self, seconds: float = 30.0) -> PlayResult:
        self._start_game()
        result = PlayResult(seconds=seconds)
        self._start_lives = self._state().get("lives") or 3
        end = time.time() + seconds
        while time.time() < end:
            t0 = time.time()
            st = self._state()
            if st.get("gameState") != 1:
                if st.get("gameState") == 2:
                    result.error = f"game over (score={st.get('score')}, lives={st.get('lives')})"
                    break
                time.sleep(0.2)
                continue
            result.ticks += 1
            targets = self._targets(st)
            cs = self._compact_state(st, targets)

            # FOCUS FIRE: clear one column bottom-to-top, then move to the
            # nearest remaining column. The fleet drifts, so column x-keys
            # change — re-adopt the nearest column to our last focus x.
            DRIFT_TOL = 10  # norm-x units
            cols = {k: v for k, v in targets.items() if k != "ufo"}

            # 0) BOTTOM ROW: a column whose lowest invader has crossed the
            # danger line overrides everything else (score, UFO, descent
            # strategy) — letting it reach the bottom is instant game over.
            # Deliberately NOT gated on reachable: if the only path to a
            # dangerously-low invader is through a bunker, shooting the
            # bunker (force_fire below) is strictly better than ignoring the
            # threat — each hit erodes the bunker tiles and may open a path
            # (or kill the invader directly) before it's too late.
            urgent = {k: v for k, v in cols.items() if v.get("lowest_y", 0) >= INVADER_DANGER_Y}
            force_fire = False

            # 1) UFO is high-value (50-300pts): track it the whole time it's
            # out, not only once it's clear of a bunker. The UFO crosses the
            # full screen in a couple seconds and often transits a bunker's
            # ~96px span along the way; waiting for "reachable" before
            # starting to chase often left too little runway to align before
            # it flew off-screen. Firing is still gated on reachability (see
            # _aim_at), so this just means the cannon is already aligned the
            # instant a gap opens.
            if urgent:
                self._current_target = max(urgent, key=lambda k: urgent[k]["lowest_y"])
                force_fire = True
            elif cs["ufo_out"]:
                self._current_target = "ufo"
            else:
                # 2) Keep the current focus column if it's still valid.
                cur = targets.get(self._current_target) if self._current_target else None
                valid = (cur is not None and cur.get("reachable")
                         and not cs["missile_below_cannon"])
                if not valid and self._current_target_x is not None and cols:
                    # re-adopt nearest column to last focus x (drift-tolerant)
                    near = min(cols.items(), key=lambda kv: abs(kv[1]["x"] - self._current_target_x))
                    if (abs(near[1]["x"] - self._current_target_x) <= DRIFT_TOL
                            and near[1].get("reachable")):
                        self._current_target = near[0]
                        cur = targets[self._current_target]
                        valid = True
                # 3) No valid focus: pick a new column.
                if not valid:
                    if self.policy == "model":
                        choice = self._decide(cs, targets)
                        result.model_calls += 1
                    else:
                        choice = self._decide_baseline(cs, targets)
                    if choice in targets:
                        self._current_target = choice
            if self._current_target in targets:
                self._current_target_x = targets[self._current_target]["x"]

            # Aim at the current target for a chunk (long enough to align +
            # fire, short enough to react to threats). Re-decide next chunk.
            if self._current_target in targets:
                result.shots += self._aim_at(self._current_target, 0.5, force_fire=force_fire)
            else:
                self._reflex_fire(st)

            # hit detection: score increase
            st2 = self._state()
            if result.score and st2.get("score") != result.score:
                result.hits += 1
            result.score = str(st2.get("score"))
            result.lives = st2.get("lives") or 0
        return result


if __name__ == "__main__":
    import argparse
    import shutil
    import tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--aggression", type=float, default=0.5,
                    help="0.0=pure survival .. 1.0=pure score (default 0.5); "
                         "scaled down automatically as lives are lost")
    ap.add_argument("--record", metavar="FILE.mp4",
                    help="record the session and save it as an mp4 (requires ffmpeg on PATH)")
    args = ap.parse_args()
    video_dir = tempfile.mkdtemp(prefix="korova_video_") if args.record else None
    with KorovaAgent(headless=not args.headless,
                     policy="baseline" if args.baseline else "model",
                     aggression=args.aggression,
                     record_video_dir=video_dir) as agent:
        r = agent.play(seconds=args.seconds)
        print(f"\nscore={r.score} lives={r.lives} ticks={r.ticks} "
              f"model_calls={r.model_calls} shots={r.shots} error={r.error!r}")
    if args.record:
        if agent.video_path:
            record_to_mp4(agent.video_path, args.record)
            print(f"recorded: {args.record}")
        else:
            print("recording failed: no video was captured")
        shutil.rmtree(video_dir, ignore_errors=True)
