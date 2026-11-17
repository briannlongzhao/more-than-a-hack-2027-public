"""Exploration: can touches be seen in the tracked ball's kinematics?

For a few 20 s windows: ball x/y, ball speed (raw and smoothed), and the team of the nearest player
with its distance. Writes data/artifact/ball_touches.png.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import features as F

f = F.load()
ball, pos, team = f["ball"], f["pos"], f["team"]
gt = np.array([l.strip().split(",")[1] for l in open(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv")).readlines()[1:]])
raw_v = np.full(len(ball), np.nan); raw_v[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * F.FPS

# Stationary ball check: frames where the ball is tracked outside the pitch (held for a throw-in)
out = ~np.isnan(ball[:, 0]) & ((ball[:, 1] < -0.5) | (ball[:, 1] > F.WIDTH + 0.5))
print("raw frame-to-frame ball speed while held out of play: median %.2f m/s, p90 %.2f m/s" % (np.nanmedian(raw_v[out]), np.nanpercentile(raw_v[out], 90)))
step = np.linalg.norm(np.diff(ball, axis=0), axis=1)
print("frame-to-frame ball step (tracked both frames): median %.2f m, p99 %.2f m, max %.1f m" % (np.nanmedian(step), np.nanpercentile(step, 99), np.nanmax(step)))
print("ball positions repeated exactly frame-to-frame: %.1f%%" % (100 * np.nanmean(step == 0)))

dn = np.linalg.norm(pos - ball[:, None, :], axis=2)
has = ~np.isnan(dn).all(1)
near = np.full(len(ball), -1); near[has] = np.nanargmin(dn[has], 1)
nd = np.full(len(ball), np.nan); nd[has] = dn[has, near[has]]

wins = [11400, 13600, 14700, 15250]
fig, axes = plt.subplots(len(wins), 1, figsize=(18, 3.6 * len(wins)))
for ax, lo in zip(axes, wins):
    hi = lo + 20 * F.FPS
    t = (np.arange(lo, hi) - lo) / F.FPS
    ax.plot(t, raw_v[lo:hi], color="0.6", lw=0.7, label="ball speed raw (m/s)")
    ax.plot(t, ball[lo:hi, 0] / 5, color="purple", lw=1, label="ball x / 5 (m)")
    ax.plot(t, ball[lo:hi, 1] / 5, color="teal", lw=1, label="ball y / 5 (m)")
    ax.set_ylim(0, 25)
    ax2 = ax.twinx(); ax2.set_ylim(0, 10); ax2.set_ylabel("nearest player dist (m)")
    for k, c in ((0, "#2b6eff"), (1, "#eb2832")):
        m = (near[lo:hi] >= 0) & (team[np.clip(near[lo:hi], 0, None)] == k)
        ax2.scatter(t[m], nd[lo:hi][m], s=3, color=c)
    g = gt[lo:hi]
    for lab, c in (("H", "#2b6eff"), ("A", "#eb2832"), ("D", "#999999")):
        ax.fill_between(t, 24, np.where(g == lab, 25, 24), step="mid", color=c, lw=0)
    ax.set_title("frames %d-%d (top strip = public GT; dots = nearest player's team & distance, right axis)" % (lo, hi), loc="left", fontsize=9)
axes[0].legend(loc="upper left", fontsize=8, ncol=3)
axes[-1].set_xlabel("seconds")
fig.tight_layout()
p = os.path.join(F.DATA_DIR, "artifact", "ball_touches.png"); fig.savefig(p, dpi=100); print(p)
