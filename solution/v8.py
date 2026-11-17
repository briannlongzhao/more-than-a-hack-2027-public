"""v8: v6 + two restart fixes found by watching the comparison clips.

  1. THE RESTART BELONGS TO THE TEAM THAT TAKES IT. v5/v6 started each live stretch with PathCRF's
     team on its first frame. PathCRF is weakest exactly there -- a throw-in taker stands off the
     pitch where the tracking often loses them -- and a throw-in clip showed 1.2 s of H before
     Athletic's two clean touches. Now a live stretch starts with the team of the first clean
     (uncontested) touch within RESTART_TOUCH s; PathCRF only if there is none. Later changes keep
     v5's discipline (>= 2 s of evidence or a clean touch).

  2. NO LIVE STRETCH SHORTER THAN MIN_LIVE BETWEEN TWO STOPPAGES. Play does not restart and stop
     again within a couple of seconds; such a gap is the stoppage detector flickering (a clip
     showed 1.1 s of H, ball not tracked, between two dead stretches). It is merged into the
     stoppage.

v7's whistle rule is not included: the reference marks the ball dead 0.6-3.8 s after the whistle.

    python solution/v8.py                 # -> evaluation/results/v8.csv
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import restart  # noqa: E402
import v1  # noqa: E402
import v5  # noqa: E402

RESTART_TOUCH = 2.0      # s after the restart in which the taker's touch is looked for
MIN_LIVE = 2 * F.FPS     # frames


def dead_mask(f, share, near, out, graded):
    dead = restart.dead_mask(f, share, near, out, graded)
    merged = 0
    for s, e in v1.runs(~dead & graded):
        if e - s < MIN_LIVE and s > 0 and dead[s - 1] and e < len(dead) and dead[e]:
            dead[s:e] = True
            merged += 1
    print("short live gaps merged into stoppages: {}".format(merged))
    return dead


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v8.csv"))
    args = p.parse_args()
    labels = v5.run(dead_fn=dead_mask, restart_touch=RESTART_TOUCH)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    v1.summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
