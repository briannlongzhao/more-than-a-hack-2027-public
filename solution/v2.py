"""v2: v1's dead-ball detection + possession driven by detected TOUCHES.

Why: the nearest player to the ball is often not the team in possession -- a pass flying past
opponents, a ball in the air over a crowd, a defender closing down. Possession should only move
when a player actually touches the ball.

  1. DEAD OR LIVE: unchanged from v1 (player speed team-wide and near the ball, ball out of pitch).

  2. TOUCHES, from two complementary signals (tools/plot_touch_detect.py shows both):
       a. kinematic: the ball's velocity over the 0.32 s after a frame differs from the 0.32 s
          before by >= DV_MIN (a kick, pass or reception), AND a player is within KICK_NEAR m --
          which rejects bounces and ball-tracking artefacts with nobody around.
       b. proximity: the nearest player's distance to the ball has a local minimum under DIP_NEAR m
          -- catches gentle receptions and dribbling, where the velocity barely changes.
     Each touch belongs to the team of the nearest player. Touches closer than MERGE frames are one
     event.

  3. POSSESSION = team of the last touch, with one exception from CHALLENGE.md ("a deflection of a
     pass that does not intercept it" is not a change): an opposing touch that is answered by the
     original team within DEFLECT s, without the opponent holding the ball for HOLD s, is a
     deflection and changes nothing. A change is placed at the touch that started it. Ball in
     flight, loose or not tracked = no touch = no change.

  4. RESTARTS: a live stretch starts with the team of its first touch; with none, the Smart Tagging
     set piece's team, else the team before the stoppage.

Thresholds were set from football reasoning before scoring and are not fitted to the public
5 minutes (RULES.md).

    python solution/v2.py                 # -> evaluation/results/v2.csv
"""

import argparse
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import v1  # noqa: E402

# ---- touches -----------------------------------------------------------------------------------
VEL_WIN = 8          # frames (0.32 s) each side for the before/after ball velocity
DV_MIN = 3.0         # m/s change in ball velocity: a kick or a reception, not tracker jitter
KICK_NEAR = 2.0      # m: a player this close confirms a velocity change as a touch
DIP_NEAR = 1.0       # m: the ball at a player's feet
PEAK = 6             # frames: a touch is the extremum within +/- this (one event, one frame)
MERGE = 6            # frames: touches closer than this are the same event

# ---- possession --------------------------------------------------------------------------------
DEFLECT = 1.0        # s: original team touches again this soon -> the opposing touch was a deflection
HOLD = 0.5           # s: ...unless the opponent kept the ball within HOLD_NEAR this long
HOLD_NEAR = 2.0      # m


def nearest_player(f):
    ball, pos, team = f["ball"], f["pos"], f["team"]
    d = np.linalg.norm(pos - ball[:, None, :], axis=2)
    has = ~np.isnan(d).all(axis=1)
    idx = np.full(len(ball), -1)
    idx[has] = np.nanargmin(d[has], axis=1)
    dist = np.full(len(ball), np.inf)
    dist[has] = d[has, idx[has]]
    near_team = np.where(idx >= 0, team[np.clip(idx, 0, None)], -1).astype(np.int8)
    return near_team, dist


def local_extreme(x, half, mode):
    pad_value = -np.inf if mode == "max" else np.inf
    win = sliding_window_view(np.pad(x, half, constant_values=pad_value), 2 * half + 1)
    return x >= win.max(axis=1) if mode == "max" else x <= win.min(axis=1)


def detect_touches(f, near_team, dist):
    """Sorted arrays (frame, team) of touch events over the whole video."""
    ball = f["ball"]
    n = len(ball)
    w = VEL_WIN
    v_before = np.full((n, 2), np.nan)
    v_after = np.full((n, 2), np.nan)
    v_before[w:] = (ball[w:] - ball[:-w]) * F.FPS / w
    v_after[:-w] = (ball[w:] - ball[:-w]) * F.FPS / w
    dv = np.nan_to_num(np.linalg.norm(v_after - v_before, axis=1))
    kick = local_extreme(dv, PEAK, "max") & (dv >= DV_MIN) & (dist <= KICK_NEAR)

    dist_f = np.where(np.isfinite(dist), dist, 1e9)
    dip = local_extreme(dist_f, PEAK, "min") & (dist_f <= DIP_NEAR)

    frames = np.flatnonzero(kick | dip)
    keep, last = [], -10 ** 9
    for fr in frames:
        if fr - last >= MERGE:
            keep.append(fr)
            last = fr
    frames = np.array(keep, dtype=np.int64)
    return frames, near_team[frames], dist


def possession(touch_frames, touch_teams, dist, near_team, live, set_pieces):
    team = np.full(len(live), -1, dtype=np.int8)
    last_team = F.HOME
    deflect, hold = int(DEFLECT * F.FPS), int(HOLD * F.FPS)
    for s, e in v1.runs(live):
        lo, hi = np.searchsorted(touch_frames, [s, e])
        tf, tt = touch_frames[lo:hi], touch_teams[lo:hi]
        cur = tt[0] if len(tt) else v1.restart_team_from_tags(s, set_pieces, last_team)
        out = np.full(e - s, cur, dtype=np.int8)
        for k in range(len(tf)):
            t, who = tf[k], tt[k]
            if who == cur:
                continue
            # opposing touch: deflection if the original team answers quickly and the opponent
            # never held the ball
            answered = k + 1 < len(tf) and tt[k + 1] == cur and tf[k + 1] - t <= deflect
            window = slice(t, min(e, t + deflect))
            held = np.sum((near_team[window] == who) & (dist[window] <= HOLD_NEAR)) >= hold
            if answered and not held:
                continue
            cur = who
            out[t - s:] = cur
        team[s:e] = out
        last_team = cur
    return team


def run():
    f = F.load()
    graded = F.graded_mask()
    share, near, _, out = v1.signals(f)
    dead = v1.dead_mask(share, near, out, graded)
    live = graded & ~dead

    near_team, dist = nearest_player(f)
    tf, tt, _ = detect_touches(f, near_team, dist)
    team = possession(tf, tt, dist, near_team, live, F.set_pieces())

    labels = np.full(F.N_FRAMES, "X", dtype="<U1")
    labels[dead] = "D"
    labels[live & (team == F.HOME)] = "H"
    labels[live & (team == F.AWAY)] = "A"
    in_play = live[tf]
    print("touches in live play: {} ({:.1f} per live minute)".format(
        int(in_play.sum()), in_play.sum() / (live.sum() / F.FPS / 60)))
    return labels


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v2.csv"))
    args = p.parse_args()
    labels = run()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    v1.summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
