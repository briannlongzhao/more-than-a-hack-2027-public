"""Exploration: is speed NEAR THE BALL a better dead-ball signal than the team-wide average?

Compares, on the same 3-minute windows as plot_speed.py:
  - share of ALL tracked players above 2 m/s
  - mean speed of the K players nearest the ball (last known ball position while it is missing)
  - ball speed from its own tracked positions (only where tracked)

Writes data/artifact/near_ball_zoom.png.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, GAMES  # noqa
import plot_speed as ps  # noqa

K = 4
FPS = GAMES[GAME]["fps"]
L, W = GAMES[GAME]["pitch"]
OUT = os.path.join(DATA_DIR, "artifact")

speed, ball_m = ps.load_speed_and_ball()
a = np.load(os.path.join(OUT, "cache", "overlay_arrays.npz"))
players = a["players"] * np.array([L, W], np.float32)
last_ball = a["last_ball"] * np.array([L, W], np.float32)
gt = a["gt"]

# K nearest to the (last known) ball
d = np.linalg.norm(players - last_ball[:, None, :], axis=2)
d = np.where(np.isnan(d), np.inf, d)
order = np.argsort(d, axis=1)[:, :K]
near_sp = np.take_along_axis(speed, order, axis=1)
near_ok = np.take_along_axis(d, order, axis=1) < np.inf
near_sp = np.where(near_ok, near_sp, np.nan)
with np.errstate(invalid="ignore"):
    near_mean = np.nanmean(near_sp, axis=1)
    frac_run = (speed > ps.RUN).sum(1) / np.maximum((~np.isnan(speed)).sum(1), 1)
frac_run = np.where((~np.isnan(speed)).sum(1) > 0, frac_run, np.nan)

# ball speed: displacement over 5 frames, only where both ends are tracked
bs = np.full(len(ball_m), np.nan, np.float32)
step = 5
bs[step:] = np.linalg.norm(ball_m[step:] - ball_m[:-step], axis=1) * FPS / step
bs = np.clip(bs, 0, 40)

np.savez(os.path.join(OUT, "cache", "near_ball.npz"), near_mean=near_mean, ball_speed=bs)

zooms = [(11185, "public window start (has GT)"), (15500, "public window, has GT"),
         (24800, "throw-ins"), (45000, "goal"), (128500, "ball out -> corner")]
fig, axes = plt.subplots(len(zooms), 1, figsize=(18, 3.3 * len(zooms)))
for ax, (lo, name) in zip(axes, zooms):
    hi = lo + 180 * FPS
    t = np.arange(hi - lo) / FPS
    ax.plot(t, ps.rolling(frac_run[lo:hi]) * 6, color="#e07b00", lw=1, label="share of ALL players > 2 m/s (x6)")
    ax.plot(t, ps.rolling(near_mean[lo:hi]), color="#2ca02c", lw=1.2, label="mean speed of %d nearest to ball (m/s)" % K)
    ax.plot(t, ps.rolling(bs[lo:hi]) / 3, color="k", lw=0.8, alpha=0.7, label="ball speed / 3 (m/s, tracked only)")
    ax.set_ylim(0, 7)
    g = gt[lo:hi]
    cols = {"H": "#2b6eff", "A": "#eb2832", "D": "#999999"}
    for lab, c in cols.items():
        ax.fill_between(t, 6.6, np.where(g == lab, 7, 6.6), step="mid", color=c, lw=0)
    miss = np.isnan(ball_m[lo:hi, 0])
    ax.fill_between(t, 0, np.where(miss, 0.25, 0), step="mid", color="#d62728", lw=0)
    for e in ps.load_smart_tagging.load():
        if e["code"] == "SET PIECES" and lo <= e["start_frame"] < hi:
            ax.axvline((e["start_frame"] - lo) / FPS, color="k", lw=0.7, alpha=0.5)
    ax.set_title("frames %d-%d: %s   (top strip = public GT where available, red base strip = ball missing)" % (lo, hi, name), loc="left", fontsize=9)
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
axes[-1].set_xlabel("seconds")
fig.tight_layout()
p = os.path.join(OUT, "near_ball_zoom.png")
fig.savefig(p, dpi=100)
print(p)
