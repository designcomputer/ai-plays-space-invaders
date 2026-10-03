# ai_invaders — Jev-style Space Invaders agent (korovatron)

A real-time browser-game agent in the **Jev / System One** style: a small
decision model makes a few high-level *constrained* choices (which column to
focus on), while a deterministic Python loop does the fast observe→aim→fire
work. It plays [korovatron.co.uk's Space Invaders](https://www.korovatron.co.uk/spaceinvaders/).

This is a **self-contained runnable copy** of the agent from the larger
`~/ai_browser` project (only the files needed to run are here).

## Run it

```bash
python example_korova.py                       # model policy, headed, 60s (watch it play)
python example_korova.py --seconds 120          # longer run
python example_korova.py --baseline             # deterministic policy (no model)
python example_korova.py --headless             # no visible browser window
python example_korova.py --aggression 0.9       # score-chasing (default 0.5)
python example_korova.py --aggression 0.0       # pure survival
```

Programmatic:

```python
from invaders_korova import KorovaAgent
with KorovaAgent(headless=False, policy="model", model="clef:27b",
                  aggression=0.5) as agent:
    r = agent.play(seconds=60)   # r.score, r.lives, r.shots, r.model_calls, r.error
```

## Files

| File | Purpose |
|------|---------|
| `invaders_korova.py` | The agent: observe (JS globals) → decide (System One) → act (deterministic aim/fire). |
| `example_korova.py` | CLI demo (headed by default). |
| `systemone.py` | The System One decision-API client (`systemone(model, state, questions, ...)`). Untouched dependency. |

External deps (already installed in system Python): `playwright` (+ Chromium), `requests`.

## Configuration

- **Decision endpoint**: `http://sriai:11434` (Ollama). Override with the
  `SYSTEMONE_BASE_URL` env var or `KorovaAgent(base_url=...)`.
- **Default model**: `clef:27b` (beats `tev1:4b` on score + survival; ~2.6×
  slower per call, 240ms vs 92ms, but the agent only calls it a handful of
  times per game). Swap with `KorovaAgent(model="tev1:4b")`.
- Other models on the server: `tev1:0.8b`, `nimble:9b`, `clef-flash:9b`,
  `granite4.2:30b`, `qwen3.6:35b`, etc.

## How it works

The korovatron game is **canvas-based**, but it exposes its full state as
**JavaScript globals**, so the agent reads state directly from JS (no canvas
pixel analysis). One `page.evaluate()` call reads:

- `gameState` (0=title, 1=playing, 1.5=lost life, 2=game over), `score`, `lives`
- `cannon.x` (cannon left edge; center = x+32; y=864)
- `fleet` — invaders `{x, y, type}` (type A=10pt, B=20pt, C=30pt)
- `activeMissile` (player bullet), `invaderMissiles` (enemy bullets)
- `shields` — 4 destructible bunkers `{x, cols[]}` (per-column open/blocked)
- `ufo` — mystery ship `{x, y}` when active

**Controls**: ArrowLeft/ArrowRight move the cannon (200 px/s), Space fires
(one bullet at a time, 750 px/s). Canvas is 896×1024.

### Architecture (constrained decision, not open-ended generation)

```
OBSERVE  read JS globals + compute per-column bunker reachability & cover
DECIDE   System One picks ONE column to focus on (or the UFO). The model only
         names a column we offered — the "model names an index we offered" pattern
AIM      deterministic: move the cannon so its center aligns with the column
         (missile fires from cannon.x+32; invader center is x+16)
REFLEX   deterministic: fire only when aligned and no bullet is active
```

**Target persistence** (drift-tolerant) keeps the focused column until it's
cleared, a UFO appears, or a missile threatens — so the model is only re-asked
a handful of times per game (~14 in a 2-minute run).

## Strategies implemented (verified against the game source)

- **Focus fire** — a bullet hits the *lowest* invader in a column first, so the
  agent clears one column bottom-to-top (10→20→20→30 = 90pts/col), then moves on.
- **Bunker reachability** — the 4 destructible bunkers block most upward shots.
  For each column the agent computes whether there's a clear vertical path
  through the bunker band, and prefers reachable columns (the map updates live
  as bunkers are broken open).
- **Slow the descent (outer columns)** — the fleet drops a row each time it
  bounces off a screen edge, bouncing off its *outermost surviving* column.
  While the fleet is wide, clear the **outer columns first** to shrink it and
  slow the descent (survival).
- **Cover awareness** — the bare screen edges have no bunker, so the agent
  prefers *covered* outer columns (a bunker beneath blocks enemy fire).
- **Precision fire** — only fire when aligned; a misaligned shot takes ~1s to
  clear the top.
- **UFO priority** — worth 50–300pts (random in this version), tracked
  continuously the instant it appears (not only once a clear shot is open —
  see below), with firing gated on three live conditions:
  1. **Bunker reachability** at the predicted aim point (below).
  2. **Fleet occlusion** — the UFO flies above the whole fleet, so *any*
     surviving invader in that column intercepts the shot long before it
     reaches the UFO. This was the dominant reason UFO shots used to whiff:
     the agent would fire, align, lead correctly, and still just hit a
     regular invader passing underneath. Early in a game, when nearly every
     column still has invaders, few UFO shots will connect — that's
     expected, not a bug; it gets easier as the fleet thins out.
  3. **Lead prediction** — the UFO moves at ~2x the cannon's speed, and a
     bullet takes ~1s to reach its altitude, so a shot aimed at its *current*
     x misses by hundreds of pixels. The game exposes the UFO's exact
     velocity as `ufo.speed` (used directly — numerically differencing
     sampled positions was tried first and was too noisy, since per-tick
     position deltas are tiny relative to `page.evaluate()` round-trip
     jitter). The agent computes time-to-altitude from `(cannon_y - ufo_y) /
     missile_speed` and aims at the UFO's predicted position at that time.
- **Reactive dodge** — `_aim_at` checks for an incoming enemy missile on the
  cannon's *current* x every 20ms tick (not just once per ~0.5s decision
  chunk) and breaks off aiming immediately to move clear. The shield band's
  bottom edge sits only ~32px above the cannon, so there's little margin once
  a missile clears the bunkers — the dodge triggers earlier, while the
  missile is still well above the shields (`dodge_trigger_y`, default y=600).
- **Aggression knob** (`aggression=0.0..1.0`, default 0.5) — tunes the
  score/survival tradeoff from the baseline's "Known tradeoff" below. Higher
  aggression raises the fleet-width threshold before the agent commits to the
  outer-column descent-slowing strategy, and once focus-firing, weighs a
  column's remaining value (invaders left = points left) against travel
  distance instead of always taking the nearest one. Effective aggression is
  scaled by `lives_remaining / starting_lives`, so even `aggression=1.0`
  automatically falls back to safe play once down to the last life. The
  model policy receives `aggression` and `life_frac` in its state too, with
  instructions to weigh risk the same way.

### Classic-arcade exploits that do NOT apply here
- **UFO shot-counting (300pt trick)**: `getRandomUfoScore()` is random, not
  shot-count based.
- **Wall of Death / Nagagoya**: invaders reaching the bottom = instant game
  over here, not the original's invulnerability bug.

## Current state / results

- Plays a **full 2-minute game and survives** (e.g. 310 pts, 1 life left at
  120s; model still alive at 90s in headless tests).
- 45s headless (median of 3): model ~130–160, baseline ~200; both survive.
- **Known tradeoff**: the descent-slowing strategy trades some score for
  survival. Aggressive focus-fire scores higher over short windows but risks
  dying sooner; the survival-first balance outscores it over longer games.
  The `--aggression` knob (above) now exposes this tradeoff directly instead
  of it being fixed — set it per run, and it backs off automatically as lives
  are lost.

## Not included here (in `~/ai_browser`)

- `korova_*.js` — reverse-engineered game source (main/Shield/Cannon/Invader/
  Missile/Ufo). Useful reference for the exact mechanics.
- `inspect_korova.py` — the one-off probe used to discover the game structure.
- The elgooG DOM-game agent (`invaders_agent.py`), the browser/monitoring
  agents, and the rest of the `systemone` client project.
