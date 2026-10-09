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

A 23-second clip of the agent playing, with the game's sound. Click to play:

https://github.com/user-attachments/assets/1cec175c-6c9a-423a-aa29-984fef88ece3

Download copy: [docs/demo.mp4](docs/demo.mp4)

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
FIRE     deterministic: hold fire while aligned and the path is clear
DODGE    deterministic: break off aiming when an enemy missile is on course
```

The model only chooses among columns the code offers, so its output is always
a valid target. Most turns the model is not consulted at all: the agent keeps
its current column until it is cleared, blocked, or threatened.

### How the game is lost

In practice the agent loses because the invaders land, not because it runs
out of lives. Each time the fleet bounces off a screen edge it drops 32px, and
the game ends when an invader reaches y=864. Wave 1's bottom row starts at
y=448, which gives 13 drops. Clearing a bottom row adds 2 more, and a narrower
fleet bounces less often. The strategy is built around these two levers.

### Strategy

- **Edges first.** While the fleet is high, the agent clears the outer
  columns. A narrower fleet takes longer to reach a wall, so it drops less
  often.
- **Bottom row once the fleet is low.** Once the lowest row passes a warning
  line (6 drops left), the agent works along that row instead of clearing one
  column to the top.
- **Emergency override.** A column 3 drops from landing overrides everything
  else, including the UFO. The agent will fire through a bunker if that is the
  only path. If it has a life to spare, it stops dodging so it can keep firing.
- **Column labels for the model.** Each column the model can pick is tagged
  EDGE, NEAR-EDGE or BOTTOM, with a height band (high, mid, LOW, CRITICAL), a
  depth, and the travel distance from the cannon. The prompt is a short
  numbered priority list. Blocked columns are not offered.
- **End-game ambush.** At 22 or fewer invaders the fleet speeds up, and
  chasing a column gets slow and inaccurate. The cannon parks at the nearest
  clear spot that some invader will sweep over, and fires when an invader's
  predicted position will be over the muzzle after the bullet's flight time.
- **Held fire key.** The game checks whether Space is held once per frame,
  so a quick tap (down and up within a millisecond or two) usually goes
  unseen. The agent holds Space while it wants to fire, which also launches
  the next bullet on the first frame the previous one is gone. This was the
  largest single improvement: fire rate went from about 0.4 to about 1 shot
  per second.
- **Targets of opportunity.** While travelling or dodging, the cannon fires
  at any invader that passes directly overhead.
- **Bunker awareness.** Four destructible bunkers block shots. Clearance is
  checked where the bullet actually travels, at the muzzle. The game removes
  the bunkers once invaders reach y=736, and the agent treats every shot as
  clear from then on.
- **UFO.** Tracked only while a shot can reach it: no bunker and no surviving
  invader in the way. Lead is computed from the UFO's exposed velocity.
  Ignored once the fleet is low.
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

Current results on a local server, 5 games per model, 300s cap, default
aggression. Each wave has 55 invaders. Kills measure progress better than
score, which UFO hits inflate.

| Model | Cleared wave 1 | Levels reached | Mean kills | Kill range | Mean score | Mean lives left | Model calls / game | Response time |
|---|---|---|---|---|---|---|---|---|
| clef:27b | 4 of 5 | 4, 5, 1, 5, 3 | 183 | 47–264 | 3348 | 2.8 | ~43 | ~345 ms |
| nimble:9b | 5 of 5 | 3, 2, 2, 3, 4 | 146 | 102–210 | 2632 | 2.8 | ~33 | ~91 ms |
| clef-flash:9b | 5 of 5 | 2, 3, 3, 3, 2 | 132 | 96–158 | 2522 | 2.6 | ~33 | ~252 ms |
| tev1:4b | 5 of 5 | 2, 3, 3, 2, 2 | 119 | 98–153 | 2208 | 2.6 | ~25 | ~90 ms |

Response time is the median of 10 decision calls with the agent's real prompt
and a captured game state, measured from the client to the model server over
the local network, with the model already loaded. The first call after a model
loads takes 4–6 s; the agent keeps the model loaded for 30 minutes, so this
happens at most once per game. The game keeps running while the model decides,
so at about 43 decisions per game clef:27b spends roughly 15 s waiting on the
model, against about 3 s for nimble:9b.

Larger models tended to go further, but the ranges overlap and clef:27b was
the most variable: it reached level 5 twice and was also the only model to
lose in wave 1. Every game that ended before the 300s cap ended with the
invaders landing, not with the agent out of lives.

Before the held-fire-key fix, clef:27b averaged 44.3 kills (range 39–49, 10
games) with the same strategy and never cleared wave 1.

Earlier results, from a previous version of the agent with a different
prompt and strategy (five runs each, score only):

| Model | Mean score | Range | Mean lives left | Model calls / game |
|---|---|---|---|---|
| clef:27b | 1082 | 770–1290 | 2.2 | ~37 |
| tev1:4b | 894 | 740–1170 | 2.8 | ~78 |
| nimble:9b | 922 | 580–1310 | 2.8 | ~41 |

Small samples; treat these as a rough guide. `clef-flash:9b` is missing from
the earlier table because it returned a server-side error (`non-finite
logit`) on every request at the time. A fix to Ollama's Windows version
resolved it.

## Files

| File | Purpose |
|---|---|
| `space_invaders_agent.py` | The agent: observe, decide, aim, fire, dodge. |
| `play_demo.py` | Command-line demo. Headed by default. |
| `systemone.py` | Minimal client for the System One decision endpoint. |

## Notes and limitations

- The game is a third-party page, and this code reads its JavaScript globals.
  If the site changes its internals, the agent will need updating.
- The landing line, drop size and shield cutoff come from the game's
  `main.js`. The warning and danger lines (`INVADER_WARN_Y`,
  `INVADER_DANGER_Y`), the ambush threshold (`AMBUSH_FLEET`) and the dodge
  thresholds are tuned heuristics.
- Tested on Windows 11 with Chromium via Playwright.

## Credits

Space Invaders was originally created by Tomohiro Nishikado at Taito (1978).
This browser version was written by Neil Kendall, who owns the website
[korovatron.co.uk](https://www.korovatron.co.uk/spaceinvaders/) where it is played. This project only plays the game and does not include or
redistribute its code or assets. Please visit the site and support the author's work.
