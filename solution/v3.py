"""v3: v2's touches, but possession only changes on CONTROLLED, UNCONTESTED touches.

v2 switched on any single opposing touch unless the original team answered quickly, and full-match
diagnostics showed it fragmented possessions more than v1 (more sub-2 s H->A->H flip-flops). Two
changes, both from CHALLENGE.md's definition that a change needs *controlled* possession:

  1. CONTESTED TOUCHES DO NOT COUNT. If players of BOTH teams are within CONTEST_NEAR m of the
     ball at a touch, the tracking cannot say who touched it (player positions are good to roughly
     a metre). Such a touch neither changes nor confirms possession.

  2. THE NEW TEAM MUST SHOW CONTROL. An opposing touch starts a *pending* change. It is confirmed
     when that team touches again before the current team does (reception + pass/dribble), or
     keeps the ball within HOLD_NEAR m for HOLD s. If the current team touches first, the pending
     change is dropped -- that was a deflection or a lost duel. A confirmed change is placed at the
     new team's FIRST touch, so timing stays sharp.

Dead-ball detection and touch detection are unchanged (v1, v2). Thresholds set from football
reasoning before scoring, not fitted to the public 5 minutes (RULES.md).

    python solution/v3.py                 # -> evaluation/results/v3.csv
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

CONTEST_NEAR = 1.5   # m: both teams this close to the ball -> contested touch
HOLD = 0.5           # s: keeping the ball this long is control on its own
HOLD_NEAR = 2.0      # m
HOLD_WINDOW = 1.0    # s after the touch in which the hold is looked for


def team_min_dist(f, frames):
    """(len(frames), 2): distance from the ball to the nearest home and nearest away player."""
    pos, ball, team = f["pos"][frames], f["ball"][frames], f["team"]
    d = np.linalg.norm(pos - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    return np.stack([d[:, team == F.HOME].min(axis=1), d[:, team == F.AWAY].min(axis=1)], axis=1)


def possession(touch_frames, touch_teams, contested, dist, near_team, live, set_pieces):
    team = np.full(len(live), -1, dtype=np.int8)
    last_team = F.HOME
    hold, hold_win = int(HOLD * F.FPS), int(HOLD_WINDOW * F.FPS)
    for s, e in v1.runs(live):
        lo, hi = np.searchsorted(touch_frames, [s, e])
        keep = ~contested[lo:hi]
        tf, tt = touch_frames[lo:hi][keep], touch_teams[lo:hi][keep]
        cur = tt[0] if len(tt) else v1.restart_team_from_tags(s, set_pieces, last_team)
        out = np.full(e - s, cur, dtype=np.int8)
        pending = None                       # frame of the challenger's first touch
        for t, who in zip(tf, tt):
            if who == cur:
                pending = None               # current team touched first: no change
                continue
            if pending is None:
                pending = t
                window = slice(t, min(e, t + hold_win))
                held = np.sum((near_team[window] == who) & (dist[window] <= HOLD_NEAR)) >= hold
                if not held:
                    continue                 # wait for a second touch
            cur = who                        # second touch, or held: control confirmed
            out[pending - s:] = cur
            pending = None
        team[s:e] = out
        last_team = cur
    return team


def run():
    f = F.load()
    graded = F.graded_mask()
    share, near, _, out = v1.signals(f)
    dead = v1.dead_mask(share, near, out, graded)
    live = graded & ~dead

    near_team, dist = v2.nearest_player(f)
    tf, tt, _ = v2.detect_touches(f, near_team, dist)
    contested = (team_min_dist(f, tf) <= CONTEST_NEAR).all(axis=1)
    team = possession(tf, tt, contested, dist, near_team, live, F.set_pieces())

    labels = np.full(F.N_FRAMES, "X", dtype="<U1")
    labels[dead] = "D"
    labels[live & (team == F.HOME)] = "H"
    labels[live & (team == F.AWAY)] = "A"
    m = live[tf]
    print("live touches {}, contested {} ({:.0%})".format(
        int(m.sum()), int((m & contested).sum()), (m & contested).sum() / max(1, m.sum())))
    return labels


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v3.csv"))
    args = p.parse_args()
    labels = run()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    v1.summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
