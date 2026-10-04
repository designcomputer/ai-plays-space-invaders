"""An AI agent that plays Space Invaders on korovatron.co.uk (headed demo).

The agent reads the game's full state from JavaScript globals (no canvas pixel
analysis), computes which columns are reachable through the four destructible
bunkers, and uses focus fire (clear one column bottom-to-top, then the next).

Usage:
    python play_demo.py                 # model policy, 60s, headed
    python play_demo.py --baseline      # deterministic policy (no model)
    python play_demo.py --seconds 30    # shorter run
"""
from __future__ import annotations

import argparse
import shutil
import tempfile

from space_invaders_agent import SpaceInvadersAgent, record_to_mp4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--baseline", action="store_true",
                    help="use the deterministic policy instead of the model")
    ap.add_argument("--headless", action="store_true",
                    help="run without a visible browser window")
    ap.add_argument("--aggression", type=float, default=0.5,
                    help="0.0=pure survival .. 1.0=pure score (default 0.5); "
                         "scaled down automatically as lives are lost")
    ap.add_argument("--model", default="clef:27b",
                    help="System One decision model, e.g. clef:27b, clef-flash:9b, tev1:4b (default clef:27b)")
    ap.add_argument("--record", metavar="FILE.mp4",
                    help="record the session and save it as an mp4 (requires ffmpeg on PATH)")
    args = ap.parse_args()

    policy = "baseline" if args.baseline else "model"
    print(f"=== Space Invaders AI — policy={policy}, model={args.model}, "
          f"{args.seconds:.0f}s, headless={args.headless}, "
          f"aggression={args.aggression} ===")
    print("Strategy: focus fire (clear a column bottom-to-top, value-weighted) "
          "+ bunker reachability + UFO priority + reactive dodge.")
    print("Watch the window: the cannon chases one column, firing through the\n"
          "bunker gaps, switches columns as it clears them, and breaks off to\n"
          "dodge incoming missiles mid-aim.\n")

    video_dir = tempfile.mkdtemp(prefix="space_invaders_video_") if args.record else None
    with SpaceInvadersAgent(headless=args.headless, policy=policy, model=args.model,
                      aggression=args.aggression, record_video_dir=video_dir) as agent:
        r = agent.play(seconds=args.seconds)

    print("\n=== RESULT ===")
    print(f"score:       {r.score}")
    print(f"level:       {r.level}")
    print(f"lives left:  {r.lives}")
    print(f"missiles:    {r.shots}")
    print(f"model calls: {r.model_calls}")
    print(f"error:       {r.error!r}")

    if args.record:
        if agent.video_path:
            record_to_mp4(agent.video_path, args.record)
            print(f"recorded:    {args.record}")
        else:
            print("recording failed: no video was captured")
        shutil.rmtree(video_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
