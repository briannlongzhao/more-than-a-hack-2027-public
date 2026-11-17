"""v5: PathCRF decides WHICH team, v3's discipline decides WHEN possession may change.

On the public window PathCRF named the right team more often than v3 (95% vs 89% of live frames),
but used alone it fragmented the full match (53 sub-2 s H->A->H flip-flops vs v3's 10). So:

  1. DEAD OR LIVE: v1's detector, unchanged.
  2. TEAM EVIDENCE: PathCRF's team per frame (solution/pathcrf_adapter.py, run first in the
     `pathcrf` env); where it has no output, v3's team.
  3. SWITCHING: inside each live stretch, a run where the evidence names the other team becomes a
     change of possession only if it is CONFIRMED -- it lasts at least MIN_RUN seconds (time to
     receive the ball and play it on), or it contains an uncontested touch by that team (v3's
     touch detector: an independent signal agreeing). Unconfirmed runs are blips and are given to
     the team in possession.

Thresholds set from football reasoning before scoring, not fitted to the public 5 minutes.

    ~/miniconda3/envs/pathcrf/bin/python solution/pathcrf_adapter.py   # once, writes the cache
    python solution/v5.py                                              # -> evaluation/results/v5.csv
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
from pathcrf_adapter import OUT as PATHCRF_CACHE  # noqa: E402

MIN_RUN = 2.0   # s


def run(dead_fn=None, restart_touch=None):
    """dead_fn(f, share, near, out, graded) -> dead mask; default is v1's detector.

    restart_touch: seconds. If set, each live stretch starts with the team of the first clean
    (uncontested) touch within that many seconds of the restart -- the team taking the set piece --
    instead of the team evidence on its first frame. Default (None) is v5's behaviour.
    """
    f = F.load()
    graded = F.graded_mask()
    share, near, _, out = v1.signals(f)
    if dead_fn is None:
        dead = v1.dead_mask(share, near, out, graded)
    else:
        dead = dead_fn(f, share, near, out, graded)
    live = graded & ~dead

    # v3 team as fallback, and v3's uncontested touches as confirmation
    near_team, dist = v2.nearest_player(f)
    tf, tt, _ = v2.detect_touches(f, near_team, dist)
    contested = (v3.team_min_dist(f, tf) <= v3.CONTEST_NEAR).all(axis=1)
    v3_team = v3.possession(tf, tt, contested, dist, near_team, live, F.set_pieces())
    clean_tf, clean_tt = tf[~contested], tt[~contested]

    if not os.path.exists(PATHCRF_CACHE):
        raise SystemExit("run solution/pathcrf_adapter.py first (pathcrf env)")
    evidence = np.load(PATHCRF_CACHE).astype(np.int8)
    evidence = np.where(evidence >= 0, evidence, v3_team)

    min_run = int(MIN_RUN * F.FPS)
    team = np.full(F.N_FRAMES, -1, dtype=np.int8)
    for s, e in v1.runs(live):
        ev = evidence[s:e]
        d = np.flatnonzero(np.diff(ev)) + 1
        starts = np.r_[0, d]
        ends = np.r_[d, e - s]
        cur = ev[0]
        if restart_touch is not None:
            lo, hi = np.searchsorted(clean_tf, [s, min(e, s + int(restart_touch * F.FPS))])
            if hi > lo:
                cur = clean_tt[lo]
        out_seg = np.empty(e - s, dtype=np.int8)
        for a, b in zip(starts, ends):
            who = ev[a]
            if who != cur:
                lo, hi = np.searchsorted(clean_tf, [s + a, s + b])
                touched = np.any(clean_tt[lo:hi] == who)
                if b - a >= min_run or touched:
                    cur = who
            out_seg[a:b] = cur
        team[s:e] = out_seg

    labels = np.full(F.N_FRAMES, "X", dtype="<U1")
    labels[dead] = "D"
    labels[live & (team == F.HOME)] = "H"
    labels[live & (team == F.AWAY)] = "A"
    return labels


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v5.csv"))
    args = p.parse_args()
    labels = run()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    v1.summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
