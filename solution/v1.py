"""v1: speed-based dead-ball detection + possession that only changes on sustained control.

Pipeline, all rules, no training:

  1. DEAD OR LIVE, independent of whether the ball is tracked.
     A frame is "still" when few players anywhere are running AND the players nearest the ball are
     walking. Still stretches of a few seconds are stoppages. The ball being tracked outside the
     pitch is a stoppage too. Players react to the whistle late and restart before they speed up,
     so each stoppage is shifted earlier by fixed reaction times.

  2. LIVE: WHICH TEAM.
     Evidence on a frame = the team of the nearest player, but only when the ball is tracked and
     that player is within CONTROL_RADIUS. Possession changes only after the other team has
     SWITCH_FRAMES of evidence in a row (a touch or deflection is not a change -- CHALLENGE.md).
     The change is back-dated to where that run of evidence began. Ball missing = no evidence =
     the current team keeps it.

  3. RESTARTS. Each live stretch starts with the team of its first evidence. With none at all,
     the Smart Tagging set piece's team, else the team before the stoppage.

Every threshold below was set from football reasoning BEFORE scoring, and is not fitted to the
public 5 minutes (RULES.md forbids tuning on them; EVALUATION.md shows why it misleads).

    python solution/v1.py                 # -> evaluation/results/v1.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402

# ---- 1. dead-ball detection --------------------------------------------------------------------
RUN_SPEED = 2.0          # m/s. Above this a player is jogging or faster; walking is ~1.4 m/s
SMOOTH = F.FPS           # 1 s centred rolling mean on the speed signals
STILL_SHARE = 0.15       # fewer than ~3 of ~22 tracked players jogging
NEAR_K = 4               # players nearest the (last known) ball
NEAR_STILL = 1.5         # m/s: those players at walking pace or slower
MIN_DEAD = 3 * F.FPS     # a stoppage lasts at least 3 s; shorter lulls are slow live play
FILL_LIVE = 1 * F.FPS    # a <1 s burst of movement inside a stoppage does not end it
OUT_MARGIN = 0.5         # m beyond a line before a tracked ball counts as out of the pitch
OUT_MERGE = 3 * F.FPS    # an out-of-pitch ball within 3 s of a stoppage belongs to it
LEAD_START = int(1.5 * F.FPS)  # players slow down ~1.5 s after the ball goes dead
LEAD_END = 2 * F.FPS           # ...and speed up ~2 s after the restart

# ---- 2. possession -----------------------------------------------------------------------------
CONTROL_RADIUS = 3.0     # m. Nearest player this close to the tracked ball = evidence of control
SWITCH_FRAMES = 20       # 0.8 s of uninterrupted opposing evidence before possession changes


def rolling_nanmean(x, w):
    v = np.nan_to_num(x)
    ok = (~np.isnan(x)).astype(np.float64)
    k = np.ones(w)
    num, den = np.convolve(v, k, "same"), np.convolve(ok, k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def runs(mask):
    """(start, end) half-open intervals where mask is True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def forward_fill(a):
    """Forward-fill NaN rows of an (n, 2) array."""
    out = a.copy()
    idx = np.where(~np.isnan(a[:, 0]), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out[:] = a[idx]
    return out


def signals(f):
    pos, speed, ball = f["pos"], f["speed"], f["ball"]
    tracked = (~np.isnan(speed)).sum(axis=1)
    share = np.where(tracked > 0, (speed > RUN_SPEED).sum(axis=1) / np.maximum(tracked, 1), np.nan)

    last_ball = forward_fill(ball)
    d = np.linalg.norm(pos - last_ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(d, axis=1)[:, :NEAR_K]
    near_sp = np.take_along_axis(speed, order, axis=1)
    near_sp[np.take_along_axis(d, order, axis=1) == np.inf] = np.nan
    with np.errstate(invalid="ignore"):
        near = np.nanmean(near_sp, axis=1)

    # nearest player to the CURRENT ball, for possession evidence
    dn = np.linalg.norm(pos - ball[:, None, :], axis=2)
    has = ~np.isnan(dn).all(axis=1)
    nearest = np.full(len(ball), -1)
    nearest[has] = np.nanargmin(dn[has], axis=1)
    near_dist = np.full(len(ball), np.inf)
    near_dist[has] = dn[has, nearest[has]]
    evidence = np.full(len(ball), -1, dtype=np.int8)
    ok = has & (near_dist <= CONTROL_RADIUS)
    evidence[ok] = f["team"][nearest[ok]]

    out = ~np.isnan(ball[:, 0]) & ((ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > F.LENGTH + OUT_MARGIN) |
                                   (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > F.WIDTH + OUT_MARGIN))
    return rolling_nanmean(share, SMOOTH), rolling_nanmean(near, SMOOTH), evidence, out


def dead_mask(share, near, out, graded):
    still = (share < STILL_SHARE) & (near < NEAR_STILL)
    still &= graded
    # close short live bursts inside stoppages, then drop short still lulls
    for s, e in runs(~still & graded):
        if e - s < FILL_LIVE and s > 0 and still[s - 1] and e < len(still) and still[e]:
            still[s:e] = True
    dead = np.zeros_like(still)
    for s, e in runs(still):
        if e - s >= MIN_DEAD:
            dead[max(0, s - LEAD_START):max(s - LEAD_START, e - LEAD_END)] = True
    # ball out of the pitch: dead, and bridge to a stoppage that is close by
    out = out & graded
    dead |= out
    for s, e in runs(~dead & graded):
        if e - s < OUT_MERGE and ((s > 0 and out[s - 1]) or (e < len(out) and out[e])):
            dead[s:e] = True
    return dead & graded


def possession(evidence, live, set_pieces, prev_default=F.HOME):
    team = np.full(len(live), -1, dtype=np.int8)
    last_team = prev_default
    for s, e in runs(live):
        ev = evidence[s:e]
        seen = np.flatnonzero(ev >= 0)
        if len(seen):
            cur = ev[seen[0]]
        else:
            cur = restart_team_from_tags(s, set_pieces, last_team)
        out = np.full(e - s, cur, dtype=np.int8)
        run_start, run_len = None, 0
        for k in range(e - s):
            t = ev[k]
            if t < 0:
                continue                      # no evidence: keep current, keep counting state
            if t == cur:
                run_start, run_len = None, 0
                continue
            if run_start is None:
                run_start = k
            run_len += 1
            if run_len >= SWITCH_FRAMES:
                cur = t
                out[run_start:] = cur          # back-date the change to where control began
                run_start, run_len = None, 0
        team[s:e] = out
        last_team = cur
    return team


def restart_team_from_tags(frame, set_pieces, fallback):
    home = F.home_team_name().lower()
    best = None
    for p in set_pieces:
        if p["team"] and abs(p["start"] - frame) < 10 * F.FPS:
            if best is None or abs(p["start"] - frame) < abs(best["start"] - frame):
                best = p
    if best is None:
        return fallback
    return F.HOME if best["team"].lower() == home else F.AWAY


def run():
    f = F.load()
    graded = F.graded_mask()
    share, near, evidence, out = signals(f)
    dead = dead_mask(share, near, out, graded)
    live = graded & ~dead
    team = possession(evidence, live, F.set_pieces())

    labels = np.full(F.N_FRAMES, "X", dtype="<U1")
    labels[dead] = "D"
    labels[live & (team == F.HOME)] = "H"
    labels[live & (team == F.AWAY)] = "A"
    return labels


def summary(labels):
    graded = F.graded_mask()
    g = labels[graded]
    n_poss, durs = 0, []
    for lab in "HA":
        for s, e in runs(labels == lab):
            n_poss += 1
            durs.append((e - s) / F.FPS)
    stops = runs(labels == "D")
    print("graded frames: H {:.1%}  A {:.1%}  D {:.1%}".format(
        np.mean(g == "H"), np.mean(g == "A"), np.mean(g == "D")))
    print("possessions {}  mean {:.1f} s  median {:.1f} s | stoppages {}  mean {:.1f} s".format(
        n_poss, np.mean(durs), np.median(durs), len(stops),
        np.mean([(e - s) / F.FPS for s, e in stops])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v1.csv"))
    args = p.parse_args()
    labels = run()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
