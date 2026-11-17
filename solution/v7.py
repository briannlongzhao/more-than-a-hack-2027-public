"""v7: v6 + the referee's stopping whistle starts the stoppage.

A stopping whistle makes the ball dead at once, but players only slow down a second or two later,
which is all v6's detector can see. Rule: a referee whistle (solution/whistle.py, strict) heard
during live play and followed by a v6 stoppage within FOLLOW s moves that stoppage's start back to
the whistle. A whistle with no stoppage after it changes nothing -- it is a crowd whistle, a
missed detection, or advantage played -- and whistles during a stoppage (restarts) change nothing
either: the ball stays dead until the kick (v6).

    python solution/whistle.py            # once: audio -> data/cache/whistles.npz
    python solution/v7.py                 # -> evaluation/results/v7.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import restart  # noqa: E402
import v1  # noqa: E402
import v5  # noqa: E402
import whistle  # noqa: E402

FOLLOW = 5.0   # s: the stoppage must start this soon after the whistle


def dead_with_whistles(f, share, near, out, graded):
    dead = restart.dead_mask(f, share, near, out, graded)
    onsets = whistle.strict_onsets()
    starts = np.array([s for s, _ in v1.runs(dead)])
    moved, total = 0, 0
    for w in onsets:
        if not graded[w] or dead[w]:
            continue
        later = starts[(starts > w) & (starts <= w + int(FOLLOW * F.FPS))]
        if len(later):
            dead[w:later[0]] = True
            moved += 1
            total += later[0] - w
    print("whistles that start a stoppage: {} (moved earlier by {:.1f} s on average)".format(
        moved, total / max(1, moved) / F.FPS))
    return dead


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v7.csv"))
    args = p.parse_args()
    labels = v5.run(dead_fn=dead_with_whistles)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    v1.summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
