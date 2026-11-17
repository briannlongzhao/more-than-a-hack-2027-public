"""v9: v8 + the referee's stopping whistle, when nobody plays on after it.

A stopping whistle kills the ball at once, but players may keep RUNNING afterwards -- towards an
injured player, the referee, a scuffle -- so the speed-based detector only confirms the stoppage
seconds later, or not at all for a while. What players do NOT do after a real stopping whistle is
play the ball on.

Rule: a referee whistle (solution/whistle.py, strict) heard during live play starts the stoppage
at the whistle when
  - v8 confirms a stoppage within FOLLOW s, and
  - no clean (uncontested) touch happens between PLAY_ON_GRACE s after the whistle and that
    stoppage -- if the ball is played on, it was not a stopping whistle (crowd, or advantage).

v7 applied whistles without the play-on check and moved one stoppage 3.8 s too early where play
had in fact continued (two clean touches after the whistle).

    python solution/v9.py                 # -> evaluation/results/v9.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
import v5  # noqa: E402
import v8  # noqa: E402
import whistle  # noqa: E402

FOLLOW = 8.0          # s: players may keep running for a while after an injury whistle
PLAY_ON_GRACE = 0.5   # s: a touch in the first half-second can be the foul itself


def dead_mask(f, share, near, out, graded):
    dead = v8.dead_mask(f, share, near, out, graded)
    near_team, dist = v2.nearest_player(f)
    tf, tt, _ = v2.detect_touches(f, near_team, dist)
    clean = tf[~(v3.team_min_dist(f, tf) <= v3.CONTEST_NEAR).all(axis=1)]
    starts = np.array([s for s, _ in v1.runs(dead)])
    used, played_on, total = 0, 0, 0
    for w in whistle.strict_onsets():
        if not graded[w] or dead[w]:
            continue
        later = starts[(starts > w) & (starts <= w + int(FOLLOW * F.FPS))]
        if not len(later):
            continue
        if np.any((clean > w + int(PLAY_ON_GRACE * F.FPS)) & (clean < later[0])):
            played_on += 1
            continue
        dead[w:later[0]] = True
        used += 1
        total += later[0] - w
    print("stopping whistles used: {} (stoppage starts {:.1f} s earlier on average); "
          "ignored because play continued: {}".format(used, total / max(1, used) / F.FPS, played_on))
    return dead


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v9.csv"))
    args = p.parse_args()
    labels = v5.run(dead_fn=dead_mask, restart_touch=v8.RESTART_TOUCH)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    v1.summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
