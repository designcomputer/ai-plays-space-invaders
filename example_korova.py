"""Watch the Jev-style agent play korovatron.co.uk Space Invaders (headed).

The agent reads the game's full state from JavaScript globals (no canvas pixel
analysis), computes which columns are reachable through the four destructible
bunkers, and uses focus fire (clear one column bottom-to-top, then the next).

Usage:
    python example_korova.py                 # model policy, 60s, headed
    python example_korova.py --baseline      # deterministic policy (no model)
    python example_korova.py --seconds 30    # shorter run
"""
from __future__ import annotations

import argparse

from invaders_korova import KorovaAgent


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
    args = ap.parse_args()

    policy = "baseline" if args.baseline else "model"
    print(f"=== korovatron Space Invaders — policy={policy}, "
          f"{args.seconds:.0f}s, headless={args.headless}, "
          f"aggression={args.aggression} ===")
    print("Strategy: focus fire (clear a column bottom-to-top, value-weighted) "
          "+ bunker reachability + UFO priority + reactive dodge.")
    print("Watch the window: the cannon chases one column, firing through the\n"
          "bunker gaps, switches columns as it clears them, and breaks off to\n"
          "dodge incoming missiles mid-aim.\n")

    with KorovaAgent(headless=args.headless, policy=policy,
                      aggression=args.aggression) as agent:
        r = agent.play(seconds=args.seconds)

    print("\n=== RESULT ===")
    print(f"score:       {r.score}")
    print(f"lives left:  {r.lives}")
    print(f"missiles:    {r.shots}")
    print(f"model calls: {r.model_calls}")
    print(f"error:       {r.error!r}")


if __name__ == "__main__":
    main()
