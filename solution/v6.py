"""v6: v5 (PathCRF team + v3 switching discipline) with stoppages ending at the restart kick.

The only change from v5 is the dead-ball END: see solution/restart.py. A corner, free kick, goal
kick or throw-in stays D until it is taken.

PathCRF's cache was computed on v1's in-play episodes; frames that v6 makes live but v1 had dead
fall back to v3's team inside v5's logic, so no rerun of the adapter is needed.

    python solution/v6.py                 # -> evaluation/results/v6.csv
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

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v6.csv"))
    args = p.parse_args()
    labels = v5.run(dead_fn=restart.dead_mask)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    v1.summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
