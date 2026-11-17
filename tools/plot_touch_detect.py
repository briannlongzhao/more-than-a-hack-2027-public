"""Exploration: touch candidates from ball-velocity change, confirmed by player proximity.

dv(t) = |v_after - v_before|, each velocity over W frames (both ends tracked). Peaks of dv above
DV_MIN are candidate touches; a candidate is CONFIRMED when a player is within NEAR_M of the ball
at that moment. Writes data/artifact/touch_detect.png and prints full-match counts.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import features as F

W = 8          # frames (0.32 s) for the before/after velocity
DV_MIN = 3.0   # m/s change in velocity to call a candidate
PEAK = 6       # frames: candidate must be the max dv within +/- this
NEAR_M = 2.0   # m: a player this close at the candidate confirms a touch

f = F.load()
ball, pos, team = f["ball"], f["pos"], f["team"]
n = len(ball)
vb = np.full((n, 2), np.nan); va = np.full((n, 2), np.nan)
vb[W:] = (ball[W:] - ball[:-W]) * F.FPS / W
va[:-W] = (ball[W:] - ball[:-W]) * F.FPS / W
dv = np.linalg.norm(va - vb, axis=1)
turn = np.degrees(np.arccos(np.clip(np.sum(va * vb, 1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-9), -1, 1)))

dvz = np.nan_to_num(dv)
from numpy.lib.stride_tricks import sliding_window_view
pad = np.pad(dvz, PEAK)
locmax = dvz >= sliding_window_view(pad, 2 * PEAK + 1).max(1)
cand = np.flatnonzero(locmax & (dvz >= DV_MIN))

dn = np.linalg.norm(pos - ball[:, None, :], axis=2)
has = ~np.isnan(dn).all(1)
near = np.full(n, -1); near[has] = np.nanargmin(dn[has], 1)
nd = np.full(n, np.nan); nd[has] = dn[has, near[has]]
conf = cand[nd[cand] <= NEAR_M]
unconf = cand[~(nd[cand] <= NEAR_M)]

graded = F.graded_mask()
live_min = graded.sum() / F.FPS / 60
print("candidates %d, confirmed %d (%.0f%%), per graded minute: %.1f confirmed" % (
    len(cand), len(conf), 100 * len(conf) / max(1, len(cand)), np.sum(graded[conf]) / live_min))
# distance dips (v2 idea from before) for comparison
dip = np.flatnonzero((np.nan_to_num(nd, nan=99) <= 1.5) & (np.nan_to_num(nd, nan=99) <= sliding_window_view(np.pad(np.nan_to_num(nd, nan=99), PEAK, constant_values=99), 2 * PEAK + 1).min(1)))
print("distance dips <=1.5 m: %d; of confirmed touches, %.0f%% have a dip within 0.3 s" % (
    len(dip), 100 * np.mean([np.any(np.abs(dip - c) <= 8) for c in conf])))
# how many changes of nearest-team at confirmed touches
tt = team[near[conf]]
print("consecutive confirmed touches by different teams: %d of %d" % (np.sum(tt[1:] != tt[:-1]), len(tt) - 1))

gt = np.array([l.strip().split(",")[1] for l in open(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv")).readlines()[1:]])
wins = [11400, 13600, 14700, 15250]
fig, axes = plt.subplots(len(wins), 1, figsize=(18, 3.8 * len(wins)))
for ax, lo in zip(axes, wins):
    hi = lo + 20 * F.FPS
    t = (np.arange(lo, hi) - lo) / F.FPS
    ax.plot(t, dv[lo:hi], color="k", lw=1, label="|velocity change| (m/s), 0.32 s before vs after")
    ax.axhline(DV_MIN, color="k", ls=":", lw=0.8)
    ax.set_ylim(0, 30)
    ax2 = ax.twinx(); ax2.set_ylim(0, 10); ax2.set_ylabel("nearest player dist (m)")
    for k, c in ((0, "#2b6eff"), (1, "#eb2832")):
        m = (near[lo:hi] >= 0) & (team[np.clip(near[lo:hi], 0, None)] == k)
        ax2.scatter(t[m], nd[lo:hi][m], s=2, color=c, alpha=0.5)
    for c in conf[(conf >= lo) & (conf < hi)]:
        col = "#2b6eff" if team[near[c]] == 0 else "#eb2832"
        ax.axvline((c - lo) / F.FPS, color=col, lw=2.2, alpha=0.85)
        ax.text((c - lo) / F.FPS, 27, " %.0f°" % turn[c], fontsize=7, color=col)
    for c in unconf[(unconf >= lo) & (unconf < hi)]:
        ax.axvline((c - lo) / F.FPS, color="0.5", lw=1.5, ls="--")
    g = gt[lo:hi]
    for lab, col in (("H", "#2b6eff"), ("A", "#eb2832"), ("D", "#999999")):
        ax.fill_between(t, 29, np.where(g == lab, 30, 29), step="mid", color=col, lw=0)
    ax.set_title("frames %d-%d   solid line = confirmed touch (team colour, turn angle), dashed grey = velocity change with nobody within %.0f m" % (lo, hi, NEAR_M), loc="left", fontsize=9)
axes[0].legend(loc="upper left", fontsize=8)
axes[-1].set_xlabel("seconds   (top strip = public GT)")
fig.tight_layout()
p = os.path.join(F.DATA_DIR, "artifact", "touch_detect.png"); fig.savefig(p, dpi=100); print(p)
np.savez(os.path.join(F.DATA_DIR, "cache", "touches_test.npz"), conf=conf, unconf=unconf, team=team[near[conf]])
