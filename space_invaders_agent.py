"""An AI agent that plays Space Invaders on korovatron.co.uk.

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

from systemone import systemone

URL = "https://www.korovatron.co.uk/spaceinvaders/"
DEFAULT_BASE_URL = os.environ.get("SYSTEMONE_BASE_URL", "http://localhost:11434")

# Canvas geometry (baseWidth=896, baseHeight=1024)
BASE_W = 896

CANNON_Y = 864
SHIELD_Y = 768
SHIELD_ROWS = 16
SHIELD_COLS = 24
TILE_PX = 4  # pixelSize = tileSize * 4

MISSILE_SPEED = 750.0  # player bullet speed, px/s
# Fleet descent (from the game's main.js): each bounce off a screen edge drops
# every invader 32px. The game ends when a drop puts an invader at y >= 864,
# and the shields switch off (shieldsOn=false: no blocking, no cover) once one
# reaches y=736. Wave 1's bottom row starts at y=448: 13 drops of headroom,
# and each bottom row cleared adds 2 more.
# Height bands for a column's lowest invader. WARN (6 drops left) switches the
# agent from clearing edge columns to clearing the bottom row; DANGER (3 drops
# left) overrides everything else, including the UFO.
INVADER_MID_Y = 576.0
INVADER_WARN_Y = 672.0
INVADER_DANGER_Y = 768.0

# End-game ambush. As the fleet shrinks it speeds up (bigger steps, shorter
# tempo) and chasing a column gets slower and less accurate. At or below
# AMBUSH_FLEET invaders (where the game raises the step size from 4 to 6px)
# the cannon parks where the fleet will pass and times each shot to the
# fleet's predicted position instead. Earlier, the fleet is too slow for
# parking to pay off and edge/bottom-row targeting matters more.
AMBUSH_FLEET = 22
FIRE_LATENCY = 0.04  # s from reading state to the bullet leaving the cannon
WALL_X = BASE_W - 32  # the fleet bounces before any invader's x passes this


def height_band(y: float) -> str:
    """How close an invader at y is to landing, as a label the model can use."""
    if y >= INVADER_DANGER_Y:
        return "CRITICAL"
    if y >= INVADER_WARN_Y:
        return "LOW"
    if y >= INVADER_MID_Y:
        return "mid"
    return "high"


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
  o.level = (typeof currentLevel !== 'undefined') ? currentLevel : null;
  o.fleetDir = (typeof Invader !== 'undefined') ? Invader.direction : null;
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
    level: int = 1  # highest wave reached (2 = cleared wave 1)


class SpaceInvadersAgent:
    def __init__(self, *, headless: bool = True, model: str = "clef:27b",
                 base_url: str | None = None, policy: str = "model",
                 keep_alive: str = "30m", aim_align: int = 4,
                 aggression: float = 0.5,
                 dodge_trigger_y: float = 600.0, dodge_radius: float = 56.0,
                 record_video_dir: str | None = None):
        # Default decision model. See the README for models compared on this game.
        self.headless = headless
        self.model = model
        self.base_url = base_url or DEFAULT_BASE_URL
        self.policy = policy  # "model" (System One) or "baseline" (deterministic)
        self.keep_alive = keep_alive
        self.aim_align = aim_align  # norm-x tolerance for "aligned"
        # 0.0 = pure survival (clear outer columns longer to slow the
        # descent), 1.0 = pure score (chase the deepest/richest column even if
        # farther away). Scaled down by remaining lives at runtime
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
        # fleet speed tracking for the ambush: (t, mean x) samples taken while
        # the fleet size and row are unchanged, plus the last good estimate
        self._fleet_track: deque[tuple[float, float]] = deque()
        self._fleet_track_key: tuple[int, float] | None = None
        self._fleet_speed_est = 0.0
        self._pw = None
        self._browser = None
        self._context = None
        self._page: Page | None = None

    # ---------- lifecycle ----------
    def __enter__(self) -> "SpaceInvadersAgent":
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
        True if no intact bunker tile sits at that x, or the shields are off."""
        if not st.get("shieldsOn", True):
            return True
        for s in st.get("shields", []):
            x0 = s["x"]
            x1 = x0 + SHIELD_COLS * TILE_PX
            if x0 - 2 <= x < x1:
                c = int((x - x0) // TILE_PX)
                if 0 <= c < SHIELD_COLS:
                    return s["cols"][c]  # open only if that column is fully destroyed
        return True  # not over any bunker -> clear

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
        high."""
        return any(inv["x"] <= x < inv["x"] + 32 for inv in st.get("fleet", []))

    def _invader_overhead(self, st: dict, cannon_x: float) -> bool:
        """Would a shot fired right now hit some invader? True if an invader
        is centered over the muzzle (cannon.x+32) and no bunker is in the
        way. Lets the cannon fire on targets of opportunity while it travels
        or dodges, instead of only at its chosen column."""
        mx = cannon_x + 32
        return (self._muzzle_clear(st, cannon_x)
                and any(abs(inv["x"] + 16 - mx) <= 10 for inv in st.get("fleet", [])))

    def _muzzle_clear(self, st: dict, cannon_x: float) -> bool:
        """Would a shot fired now pass the bunkers? The bullet is 4px wide at
        cannon.x+32, so both of its edges must be over open bunker columns."""
        mx = cannon_x + 32
        return self._reachable_x(st, mx) and self._reachable_x(st, mx + 3)

    def _ufo_reachable_x(self, st: dict, x: float) -> bool:
        """Can a bullet aimed at x actually reach the UFO's altitude? Needs
        both a clear bunker column AND no surviving invader in the way."""
        return self._reachable_x(st, x) and not self._column_has_invader(st, x)

    def _targets(self, st: dict) -> dict[str, dict]:
        """Offered target set: invader columns + UFO. Each column is annotated
        with reachability (can a bullet fired at its x get through the
        bunkers?), its position in the fleet (EDGE / NEAR-EDGE), whether it
        holds part of the fleet's bottom row (BOTTOM), and how close its
        lowest invader is to landing (height_band)."""
        targets: dict[str, dict] = {}
        # Group invaders into columns by x (they move together). For each
        # column, the lowest invader is what a bullet hits first.
        cols: dict[float, list[dict]] = {}
        for inv in st.get("fleet", []):
            cols.setdefault(round(inv["x"]), []).append(inv)
        xs = sorted(cols)
        bottom_y = max((inv["y"] for inv in st.get("fleet", [])), default=0.0)
        for i, x in enumerate(xs):
            members = cols[x]
            lowest = max(members, key=lambda m: m["y"])  # closest to player
            nx = norm(x)
            reach = self._reachable_x(st, x)
            depth = len(members)
            edge_rank = min(i, len(xs) - 1 - i)  # 0 = outermost column
            bottom = lowest["y"] >= bottom_y - 8
            band = height_band(lowest["y"])
            tags = []
            if edge_rank == 0:
                tags.append("EDGE")
            elif edge_rank == 1:
                tags.append("NEAR-EDGE")
            if bottom:
                tags.append("BOTTOM")
            tags += [band, f"{depth} deep", "open" if reach else "BLOCKED"]
            targets[f"c{nx}"] = {"x": nx, "cx": x, "reachable": reach,
                                 "depth": depth, "lowest_y": lowest["y"],
                                 "edge_rank": edge_rank, "bottom": bottom, "band": band,
                                 "label": f"col x={nx}: {', '.join(tags)}"}
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
        life_frac = self._life_frac(lives)
        eff = self.aggression * life_frac
        fleet_lowest_y = max((inv["y"] for inv in st.get("fleet", [])), default=0.0)
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
            # aggression scaled by lives left, as a word: small models handle
            # a category better than a product of two numbers
            "mode": "survival" if eff < 0.34 else ("score" if eff > 0.66 else "balanced"),
            "fleet_lowest_y": fleet_lowest_y,
            "fleet_lowest_band": height_band(fleet_lowest_y),
            "columns_left": sum(1 for k in targets if k != "ufo"),
        }

    def _life_frac(self, lives: int | None) -> float:
        """Lives remaining as a fraction of starting lives (1.0 = none lost)."""
        lives = lives or 0
        start_lives = self._start_lives or max(lives, 1)
        return max(0.0, min(1.0, lives / start_lives))

    # ---------- decide ----------
    def _decide(self, cs: dict, targets: dict[str, dict]) -> str | None:
        # The UFO is handled deterministically in play(), and dodging in
        # _aim_at, so the model only picks among columns. Blocked columns
        # waste the bullet on a bunker; leave them out when there's a choice.
        options = {k: v for k, v in targets.items() if k != "ufo"}
        open_options = {k: v for k, v in options.items() if v.get("reachable")}
        if len(open_options) >= 2:
            options = open_options
        if not options:
            return None
        if len(options) == 1:  # choice needs 2-26 candidates
            return next(iter(options))
        state = {
            "cannon_x": cs["cannon_x"],
            "columns_left": cs["columns_left"],
            "fleet_lowest_band": cs["fleet_lowest_band"],
            "mode": cs["mode"],
        }
        questions = {
            "target": {
                "type": "choice",
                "instructions": (
                    "You are playing Space Invaders and choose which invader "
                    "column the cannon shoots at. x is 0-100 (0=left, "
                    "100=right). The fleet drops one row every time its "
                    "outermost column hits a screen wall, and the game is lost "
                    "when its lowest invader lands. Height bands: high > mid > "
                    "LOW > CRITICAL (about to land).\n"
                    "Each column's dist is how far the cannon must travel to "
                    "reach it; travelling costs shooting time.\n"
                    "Pick ONE column. Priorities, in order:\n"
                    "1. A CRITICAL or LOW column: its lowest invader must not land.\n"
                    "2. While fleet_lowest_band is high: EDGE, then NEAR-EDGE "
                    "columns. A narrower fleet hits the walls, and drops, "
                    "less often.\n"
                    "   Once fleet_lowest_band is mid or lower: BOTTOM columns. "
                    "Clearing the lowest row buys time before the fleet lands.\n"
                    "3. Avoid dist above 30 unless rule 1 applies; between "
                    "similar options take the smaller dist.\n"
                    "mode=survival: follow the priorities strictly. "
                    "mode=score: you may take a deeper (more points) column "
                    "if it is close and no column is LOW or CRITICAL."
                ),
                "criteria": {k: f"{v['label']}, dist={abs(v['x'] - cs['cannon_x'])}"
                             for k, v in options.items()},
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
        always taking the nearest column.

        BOTTOM ROW: once the fleet's lowest row passes INVADER_WARN_Y, only
        columns holding part of that row are considered. Each row cleared
        buys two more drops before the fleet lands."""
        if not targets:
            return None
        cx = cs["cannon_x"]
        cols = {k: v for k, v in targets.items() if k != "ufo"}
        reachable = {k: v for k, v in cols.items() if v.get("reachable")}
        pool = reachable or cols
        if not pool:
            return None

        eff_aggression = self.aggression * self._life_frac(cs.get("lives"))

        if cs["fleet_lowest_y"] >= INVADER_WARN_Y:
            bottom = {k: v for k, v in pool.items() if v.get("bottom")}
            if bottom:
                # nearest bottom column; an edge column breaks near-ties
                return min(bottom, key=lambda k: abs(bottom[k]["x"] - cx)
                           - 5 * (bottom[k]["edge_rank"] == 0))

        # the fleet has 11 columns: clear edges until 9 (aggressive) .. 5
        # (survival) remain
        wide_threshold = 5 + round((1.0 - eff_aggression) * 4)
        if len(cols) >= wide_threshold:
            # Prioritize OUTER columns: clearing them shrinks the fleet, so it
            # bounces (and drops) less often. Prefer the outermost, then the
            # nearest. Bunker cover is ignored: a column that is both covered
            # and open sits over a narrow bunker hole and drifts out of it
            # within a second or two, leaving the cannon chasing a blocked
            # column; the reactive dodge handles enemy fire instead.
            outer = {k for k, v in pool.items() if v["edge_rank"] <= 2}  # outermost 3 per side
            if outer:
                return min(outer, key=lambda k: abs(pool[k]["x"] - cx) + 15 * pool[k]["edge_rank"])

        # Narrow fleet (or aggressive enough to skip descent-slowing): score by
        # remaining value vs. travel cost. Survival-leaning settings stay
        # nearest-column; score-leaning settings reach further for
        # a deeper, richer column. URGENCY (how close the column's lowest
        # invader is to the warning line) is weighted heavily, and BOTTOM-row
        # columns get a bonus, so the lowest invaders are thinned out before
        # the bottom-row phase above kicks in.
        def score(k: str) -> float:
            v = pool[k]
            value = v.get("depth", 0) / 4.0  # normalize (max column depth ~4-5)
            dist_cost = abs(v["x"] - cx) / 100.0
            bottom_bonus = 0.3 if v.get("bottom") else 0.0
            urgency = max(0.0, (v.get("lowest_y", 0) - INVADER_MID_Y) / (INVADER_WARN_Y - INVADER_MID_Y))
            return (eff_aggression * value - (1.0 - 0.5 * eff_aggression) * dist_cost
                    + bottom_bonus + 1.5 * urgency)

        return max(pool, key=score)

    # ---------- act ----------
    def _reflex_fire(self, st: dict) -> bool:
        if st.get("gameState") == 1 and not st.get("missile"):
            self.page.keyboard.press("Space")
            return True
        return False

    def _aim_at(self, target_key: str, duration: float, force_fire: bool = False,
                bottom_only: bool = False, dodge: bool = True) -> int:
        """Move cannon toward the live x of `target_key`, firing throughout.
        Re-reads the target's current x each iteration (the fleet drifts).

        force_fire=True skips the reachability gate on firing (still requires
        alignment) — used for the bottom-row emergency override in play(): if
        a dangerously-low invader's only path is through a bunker, shooting
        the bunker to erode it is strictly better than holding fire.

        bottom_only=True stops as soon as the target column no longer holds
        part of the fleet's bottom row, so the next shot goes to another
        bottom-row invader instead of further up the same column.

        dodge=False ignores enemy missiles: play() uses it when the fleet is
        about to land and a life can be spared, since a lost life costs less
        than a landing (game over).

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
        as `ufo.speed` (used directly, since per-tick position deltas are tiny
        relative to page.evaluate() round-trip jitter and differencing them is
        too noisy). The fleet doesn't expose a velocity field and moves in
        discrete steps rather than continuously, so we estimate it from a short
        rolling window of samples (noisy single-tick deltas average out).

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

            threat = self._threat(st, cx) if dodge else None
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
                # keep shooting while dodging: once the fleet is low, enemy
                # missiles are in the dodge band almost constantly
                if not st.get("missile") and self._invader_overhead(st, cx):
                    self.page.keyboard.press("Space")
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
            if bottom_only and not tgt.get("bottom"):
                break
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
            # Targets of opportunity: while travelling to a column, fire at
            # any invader that passes over the muzzle (not while lining up on
            # the UFO, where an early shot would hold the one bullet slot).
            # The planned path (reachable, at the predicted x) and the actual
            # muzzle path must both be clear: within the alignment tolerance
            # the muzzle can sit over a bunker edge the target column misses.
            if target_key == "ufo":
                reachable = (reachable and self._muzzle_clear(st, cx)
                             and not self._column_has_invader(st, cx + 32))
            else:
                reachable = reachable and self._muzzle_clear(st, cx)
            if not st.get("missile") and (
                    (aligned and (reachable or force_fire))
                    or (target_key != "ufo" and self._invader_overhead(st, cx))):
                self.page.keyboard.press("Space")
            if st.get("missile") and not had_missile:
                fired += 1
            had_missile = st.get("missile") is not None
            time.sleep(0.02)
        if key:
            self.page.keyboard.up(key)
        return fired

    def _fleet_speed(self, st: dict) -> float:
        """Fleet horizontal speed (px/s, magnitude) from ~0.3s of samples of
        the fleet's mean x. Samples restart whenever an invader dies (the
        mean jumps) or the fleet drops a row (it just reversed); until enough
        new samples arrive, the previous estimate is reused."""
        fleet = st.get("fleet", [])
        if not fleet:
            return self._fleet_speed_est
        now = time.time()
        key = (len(fleet), max(inv["y"] for inv in fleet))
        if key != self._fleet_track_key:
            self._fleet_track.clear()
            self._fleet_track_key = key
        self._fleet_track.append((now, sum(inv["x"] for inv in fleet) / len(fleet)))
        while now - self._fleet_track[0][0] > 0.3:
            self._fleet_track.popleft()
        t0, x0 = self._fleet_track[0]
        if now - t0 >= 0.1:
            self._fleet_speed_est = abs(self._fleet_track[-1][1] - x0) / (now - t0)
        return self._fleet_speed_est

    def _shot_will_hit(self, st: dict, cannon_x: float, speed: float) -> bool:
        """Will a bullet fired now from cannon_x meet an invader? Each
        invader's x is projected forward by the bullet's flight time to its
        row, reflecting off the wall if the fleet bounces first (the drop
        that comes with a bounce is ignored)."""
        fleet = st.get("fleet", [])
        if not fleet:
            return False
        v = speed if st.get("fleetDir") == "right" else -speed
        lo = min(inv["x"] for inv in fleet)
        hi = max(inv["x"] for inv in fleet)
        room = (WALL_X - hi) if v > 0 else lo  # travel left before the bounce
        bullet_cx = cannon_x + 34  # 4px bullet at cannon.x+32
        for inv in fleet:
            t = FIRE_LATENCY + max(0.0, CANNON_Y - (inv["y"] + 32)) / MISSILE_SPEED
            dx = abs(v) * t
            if dx > room:
                dx = room - (dx - room)
            x = inv["x"] + (dx if v > 0 else -dx)
            if abs(x + 16 - bullet_cx) <= 12:
                return True
        return False

    def _park_x(self, st: dict, cannon_x: float) -> float:
        """Nearest cannon x that some invader will pass over and whose shots
        clear the bunkers (with an 8px margin on each side so small position
        drift doesn't put the muzzle over a bunker edge).

        The fleet only sweeps between the walls, so each invader covers a
        limited band of x; once the columns over a spot are cleared, the
        fleet's gap can sweep back and forth across it without any invader
        ever passing overhead. Only spots inside some invader's band count."""
        fleet = st.get("fleet", [])
        if not fleet:
            return cannon_x
        lo = min(inv["x"] for inv in fleet)
        hi = max(inv["x"] for inv in fleet)
        # range of each invader's center, as the fleet sweeps wall to wall
        bands = [(inv["x"] - lo + 16, inv["x"] + (WALL_X - hi) + 16) for inv in fleet]
        spots = [x for x in range(0, BASE_W - 64 + 1, 4)
                 if any(a + 8 <= x + 34 <= b - 8 for a, b in bands)
                 and self._muzzle_clear(st, x - 8) and self._muzzle_clear(st, x + 8)]
        return min(spots, key=lambda x: abs(x - cannon_x), default=cannon_x)

    def _ambush(self, duration: float, dodge: bool = True) -> int:
        """End-game: park at the nearest clear spot and fire whenever the
        fleet's predicted position puts an invader over the muzzle (see
        AMBUSH_FLEET). Dodges like _aim_at. Returns missiles launched."""
        fired = 0
        had_missile = False
        key = None
        end = time.time() + duration
        while time.time() < end:
            st = self._state()
            if st.get("gameState") != 1 or not st.get("fleet"):
                break
            cx = st.get("cannonX")
            if cx is None:
                break
            speed = self._fleet_speed(st)
            threat = self._threat(st, cx) if dodge else None
            if threat is not None:
                want = "ArrowRight" if threat["x"] < cx + 32 else "ArrowLeft"
            else:
                diff = self._park_x(st, cx) - cx
                want = None if abs(diff) <= 4 else ("ArrowRight" if diff > 0 else "ArrowLeft")
            if want != key:
                if key:
                    self.page.keyboard.up(key)
                if want:
                    self.page.keyboard.down(want)
                key = want
            if (not st.get("missile") and self._muzzle_clear(st, cx)
                    and self._shot_will_hit(st, cx, speed)):
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
            level = st.get("level") or 1
            if level > result.level:
                result.level = level
                self._current_target = self._current_target_x = None  # new fleet

            # END-GAME: few invaders left and too fast to chase; park and
            # time shots instead (no column choice, so no model call).
            fleet = st.get("fleet", [])
            if 0 < len(fleet) <= AMBUSH_FLEET:
                danger = max(inv["y"] for inv in fleet) >= INVADER_DANGER_Y
                result.shots += self._ambush(
                    0.5, dodge=not (danger and (st.get("lives") or 0) > 1))
                self._current_target = self._current_target_x = None
                self._record(result)
                continue

            targets = self._targets(st)
            cs = self._compact_state(st, targets)

            # FOCUS FIRE: clear one column bottom-to-top, then move to the
            # nearest remaining column. The fleet drifts, so column x-keys
            # change — re-adopt the nearest column to our last focus x.
            # Once the fleet's bottom row passes INVADER_WARN_Y, a focus
            # column stays valid only while it still holds part of the bottom
            # row: the agent then works along that row instead of up a column.
            # (Only while some bottom column is reachable; otherwise any open
            # column beats waiting for one to leave a bunker's shadow.)
            DRIFT_TOL = 10  # norm-x units
            cols = {k: v for k, v in targets.items() if k != "ufo"}
            fleet_low = cs["fleet_lowest_y"] >= INVADER_WARN_Y
            want_bottom = fleet_low and any(v.get("bottom") and v.get("reachable")
                                            for v in cols.values())

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

            # 1) UFO is high-value (50-300pts): track it from the moment it
            # appears. It crosses the screen in a couple of seconds and may pass
            # behind a bunker along the way, so the cannon should already be
            # aligned when a gap opens. Firing is still gated on reachability
            # (see _aim_at). Only taken up while a shot can currently reach
            # it (chasing a UFO hidden behind the fleet wastes seconds of
            # firing time), and skipped once the fleet is low: survival first.
            if urgent:
                self._current_target = max(urgent, key=lambda k: urgent[k]["lowest_y"])
                force_fire = True
            elif cs["ufo_out"] and not fleet_low and targets["ufo"].get("reachable"):
                self._current_target = "ufo"
            else:
                # 2) Keep the current focus column if it's still valid.
                cur = targets.get(self._current_target) if self._current_target else None
                valid = (cur is not None and cur.get("reachable")
                         and not cs["missile_below_cannon"]
                         and (cur.get("bottom") or not want_bottom))
                if not valid and self._current_target_x is not None and cols:
                    # re-adopt nearest column to last focus x (drift-tolerant)
                    near = min(cols.items(), key=lambda kv: abs(kv[1]["x"] - self._current_target_x))
                    if (abs(near[1]["x"] - self._current_target_x) <= DRIFT_TOL
                            and near[1].get("reachable")
                            and (near[1].get("bottom") or not want_bottom)):
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
                result.shots += self._aim_at(self._current_target, 0.5, force_fire=force_fire,
                                             bottom_only=want_bottom and not force_fire
                                             and self._current_target != "ufo",
                                             dodge=not (force_fire and (st.get("lives") or 0) > 1))
            else:
                self._reflex_fire(st)
            self._record(result)
        return result

    def _record(self, result: PlayResult) -> None:
        """Update score, lives and hit count after a chunk of play."""
        st2 = self._state()
        if result.score and st2.get("score") != result.score:
            result.hits += 1
        result.score = str(st2.get("score"))
        result.lives = st2.get("lives") or 0


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
    ap.add_argument("--model", default="clef:27b",
                    help="System One decision model, e.g. clef:27b, clef-flash:9b, tev1:4b (default clef:27b)")
    ap.add_argument("--record", metavar="FILE.mp4",
                    help="record the session and save it as an mp4 (requires ffmpeg on PATH)")
    args = ap.parse_args()
    video_dir = tempfile.mkdtemp(prefix="space_invaders_video_") if args.record else None
    with SpaceInvadersAgent(headless=not args.headless,
                     policy="baseline" if args.baseline else "model",
                     model=args.model,
                     aggression=args.aggression,
                     record_video_dir=video_dir) as agent:
        r = agent.play(seconds=args.seconds)
        print(f"\nscore={r.score} level={r.level} lives={r.lives} ticks={r.ticks} "
              f"model_calls={r.model_calls} shots={r.shots} error={r.error!r}")
    if args.record:
        if agent.video_path:
            record_to_mp4(agent.video_path, args.record)
            print(f"recorded: {args.record}")
        else:
            print("recording failed: no video was captured")
        shutil.rmtree(video_dir, ignore_errors=True)
