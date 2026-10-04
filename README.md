<p align="center">
  <img src="https://www.korovatron.co.uk/images/icons/invaderIcon.png" width="96" alt="Space Invaders invader icon (Neil Kendall / Korovatron)">
</p>

# AI Plays Space Invaders

An AI agent that plays [korovatron.co.uk's Space Invaders](https://www.korovatron.co.uk/spaceinvaders/)
in a real browser. A small language model makes high-level decisions (which
column to focus on), while deterministic code handles the fast work of aiming,
firing, and dodging.

**Purpose:** this project demonstrates using open-source decision models served
through [Ollama](https://ollama.com), applied to a real-time game. The
approach is inspired by Jev's constrained-decision style, where a model picks
one option from a list the code offers rather than generating open-ended output.
The game is a demo target, not the point: the same pattern applies to any task
that needs fast control loops with occasional model input.

## Demo

A 23-second clip of the agent playing, with the game's sound:
[▶ Watch the demo (docs/demo.mp4)](docs/demo.mp4)

## Quick start

Requirements: Python 3.11+, and a [System One](#decision-model)-compatible
model server (Ollama-style endpoint). Optional: `ffmpeg` on your PATH for
`--record`.

```bash
pip install playwright requests
python -m playwright install chromium

# point at your model server (default: http://localhost:11434)
export SYSTEMONE_BASE_URL=http://localhost:11434    # PowerShell: $env:SYSTEMONE_BASE_URL="..."

python play_demo.py                         # watch the AI play in a visible window
```

## Demo options

```bash
python play_demo.py                          # model policy, headed, 60s
python play_demo.py --seconds 300            # longer run (ends early on game over)
python play_demo.py --model nimble:9b        # choose the decision model
python play_demo.py --aggression 0.9         # score-chasing (default 0.5; 0.0 = pure survival)
python play_demo.py --baseline               # deterministic policy, no model server needed
python play_demo.py --headless               # no visible window
python play_demo.py --record game.mp4        # save the session as mp4 (needs ffmpeg)
```

Video is captured silently. Playwright records only frames; to include game
audio, record the window with a system tool (e.g. Windows Game Bar,
`Win+Alt+R`) while the demo runs.

## How it works

```
OBSERVE  read game state from JavaScript globals (no pixel analysis)
DECIDE   the model picks ONE column to focus on, from options we offer it
AIM      deterministic: align the cannon, leading moving targets
FIRE     deterministic: fire when aligned, the path is clear, and no shot is in flight
DODGE    deterministic: break off aiming when an enemy missile is on course
```

The model only chooses among columns the code offers, so its output is always
a valid target. Most turns the model is not consulted at all: the agent keeps
its current column until it is cleared, blocked, or threatened.

### Strategy

- **Focus fire.** A bullet hits the lowest invader in a column first, so the
  agent clears columns bottom-to-top.
- **Bunker awareness.** Four destructible bunkers block shots. Each column is
  marked reachable or blocked, and the agent prefers reachable ones.
- **Slow the descent.** While the fleet is wide, clear the outer columns first.
  The fleet drops a row each time it bounces off a screen edge, so shrinking
  its width buys time. Covered outer columns (with a bunker beneath) are
  preferred.
- **Bottom-row urgency.** A column whose lowest invader nears the bottom
  overrides everything else, since reaching the bottom ends the game. The agent
  will fire through a bunker if that is the only path.
- **UFO.** Tracked as soon as it appears. Fired on only when a shot can reach it:
  no bunker and no surviving invader in the way. Lead is computed from the
  UFO's exposed velocity, so the shot lands where the UFO will be.
- **Fleet lead.** A bullet takes real time to reach a tall column, and the fleet
  drifts or reverses during that flight. The agent estimates fleet velocity from
  recent samples and aims at the predicted position.
- **Drift-tolerant tracking.** A column's identity can change as it drifts. The
  agent tracks the target by physical position, so a rename does not look like a
  cleared column.
- **Reactive dodge.** Checked every 20 ms, not once per decision. Missiles
  inside the danger band trigger an immediate move.
- **Aggression.** `aggression` in [0, 1] trades safety for score. Effective
  aggression scales down with remaining lives, so the agent plays safe on its
  last life.

## Decision model

The model is reached through a System One endpoint (`POST /v1/systemone`).
`systemone.py` is a small client for it. Model choice is set with `--model`
or `SpaceInvadersAgent(model=...)`.

Models tested on a local server, five runs each, 300s cap, default aggression:

| Model | Mean score | Range | Mean lives left | Model calls / game |
|---|---|---|---|---|
| clef:27b | 1082 | 770–1290 | 2.2 | ~37 |
| tev1:4b | 894 | 740–1170 | 2.8 | ~78 |
| nimble:9b | 922 | 580–1310 | 2.8 | ~41 |

Small samples; treat these as a rough guide. `clef-flash:9b` returned a
server-side error (`non-finite logit`) on every request on the test server.

## Files

| File | Purpose |
|---|---|
| `space_invaders_agent.py` | The agent: observe, decide, aim, fire, dodge. |
| `play_demo.py` | Command-line demo. Headed by default. |
| `systemone.py` | Minimal client for the System One decision endpoint. |

## Notes and limitations

- The game is a third-party page, and this code reads its JavaScript globals.
  If the site changes its internals, the agent will need updating.
- The bottom-row danger line (`INVADER_DANGER_Y`) and dodge thresholds are
  heuristics. The game's exact loss threshold is not known here.
- Tested on Windows 11 with Chromium via Playwright.

## Credits

Space Invaders was written by Neil Kendall, who also owns the website
[korovatron.co.uk](https://www.korovatron.co.uk/spaceinvaders/) where the game is played. This project only plays the game and does not include or
redistribute its code or assets. Please visit the site and support the author's work.
