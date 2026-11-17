"""Exploration: does player speed reveal dead-ball stretches?

Plots, per frame (1 s rolling mean):
  - mean speed of tracked players (m/s), and the share of them above 2 m/s (jogging or faster)
  - ball tracked / missing, and ball tracked outside the pitch lines
  - Smart Tagging set pieces (restarts) as markers
  - the public ground truth where it exists (first 5 min) -- for reading the plot, not for tuning

Writes data/artifact/speed_overview.png (both halves) and speed_zoom.png (a few 3-minute windows).

    python tools/plot_speed.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
import load_smart_tagging  # noqa: E402

OUT_DIR = os.path.join(DATA_DIR, "artifact")
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N = GAMES[GAME]["n_video_frames"]
SPEED_CACHE = os.path.join(OUT_DIR, "cache", "speed.npz")
RUN = 2.0  # m/s: above this a player is at least jogging


def load_speed_and_ball():
    if os.path.exists(SPEED_CACHE):
        z = np.load(SPEED_CACHE)
        return z["speed"], z["ball"]
    speed = np.full((N, 46), np.nan, dtype=np.float32)
    ball = np.full((N, 2), np.nan, dtype=np.float32)
    with open(asset_paths()["atd_raw"]) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] != "N":
                    s = group.split(",")[2]
                    if s[0] != "N":
                        speed[i, slot] = float(s)
            if parts[2][0] != "N":
                x, y = parts[2].split(",")[:2]
                ball[i] = (float(x) * LENGTH, float(y) * WIDTH)
    os.makedirs(os.path.dirname(SPEED_CACHE), exist_ok=True)
    np.savez(SPEED_CACHE, speed=speed, ball=ball)
    return speed, ball


def rolling(x, w=FPS):
    """Centred rolling mean ignoring NaN."""
    v = np.nan_to_num(x)
    c = np.convolve(v, np.ones(w), "same")
    n = np.convolve((~np.isnan(x)).astype(float), np.ones(w), "same")
    with np.errstate(invalid="ignore"):
        return c / n


def main():
    speed, ball = load_speed_and_ball()
    with np.errstate(invalid="ignore"):
        mean_speed = np.nanmean(speed, axis=1)
        tracked = (~np.isnan(speed)).sum(axis=1)
        frac_run = np.where(tracked > 0, (speed > RUN).sum(axis=1) / np.maximum(tracked, 1), np.nan)
    ms, fr = rolling(mean_speed), rolling(frac_run)
    missing = np.isnan(ball[:, 0])
    out = ~missing & ((ball[:, 0] < -0.5) | (ball[:, 0] > LENGTH + 0.5) |
                      (ball[:, 1] < -0.5) | (ball[:, 1] > WIDTH + 0.5))

    gt = np.load(os.path.join(OUT_DIR, "cache", "overlay_arrays.npz"))["gt"]
    events = [e for e in load_smart_tagging.load() if e["code"] == "SET PIECES"]

    halves = half_video_blocks(GAME)
    graded = np.zeros(N, bool)
    for _, lo, hi in halves:
        graded[lo:hi + 1] = True
    print("graded frames: mean speed {:.2f} m/s, share running {:.0%}".format(
        np.nanmean(mean_speed[graded]), np.nanmean(frac_run[graded])))
    print("ball tracked outside the pitch on {} graded frames".format(int(out[graded].sum())))

    def panel(ax_s, ax_b, lo, hi, minutes, title):
        f = np.arange(lo, hi)
        t = (f - lo) / FPS / (60 if minutes else 1)
        ax_s.plot(t, ms[lo:hi], color="#1f5fbf", lw=0.8, label="mean player speed (m/s)")
        ax2 = ax_s.twinx()
        ax2.plot(t, fr[lo:hi], color="#e07b00", lw=0.8, label="share > 2 m/s")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("share > 2 m/s", color="#e07b00")
        ax_s.set_ylabel("mean m/s", color="#1f5fbf")
        ax_s.set_ylim(0, max(4, np.nanmax(ms[lo:hi]) * 1.05))
        ax_s.set_title(title, loc="left", fontsize=10)
        for e in events:
            if lo <= e["start_frame"] < hi:
                x = (e["start_frame"] - lo) / FPS / (60 if minutes else 1)
                ax_s.axvline(x, color="k", lw=0.7, alpha=0.6)
                if not minutes:
                    ax_s.text(x, ax_s.get_ylim()[1] * 0.97, " " + e["tags"].get("Type", ""),
                              fontsize=7, va="top", rotation=90)
        # strips: ball missing, ball out, public GT
        rows = [("ball missing", np.where(missing[lo:hi], 1, 0), "#d62728"),
                ("ball out of pitch", np.where(out[lo:hi], 1, 0), "#9467bd")]
        for k, (name, v, col) in enumerate(rows):
            ax_b.fill_between(t, k, k + 0.8 * v, step="mid", color=col, lw=0)
            ax_b.text(t[0], k + 0.4, name + " ", ha="right", va="center", fontsize=8)
        g = gt[lo:hi]
        if (g != "X").any():
            colors = {"H": "#2b6eff", "A": "#eb2832", "D": "#999999"}
            for lab, col in colors.items():
                ax_b.fill_between(t, 2, 2 + 0.8 * (g == lab), step="mid", color=col, lw=0)
            ax_b.text(t[0], 2.4, "public GT ", ha="right", va="center", fontsize=8)
        ax_b.set_ylim(0, 3)
        ax_b.set_yticks([])
        ax_b.set_xlabel("minutes into half" if minutes else "seconds")

    # Overview: both halves
    fig, axes = plt.subplots(4, 1, figsize=(22, 11), gridspec_kw=dict(height_ratios=[3, 1, 3, 1]))
    for k, (half, lo, hi) in enumerate(halves):
        panel(axes[2 * k], axes[2 * k + 1], lo, hi + 1, True,
              "Half {} (1 s rolling mean). Black lines = Smart Tagging set pieces".format(half))
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "speed_overview.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # Zooms: 3-minute windows at moments we already looked at in the overlay previews
    zooms = [(11185, "public window start (has GT)"), (15500, "public window: goal kick at 17460"),
             (24800, "throw-in at 25410"), (45000, "goal at ~46000"),
             (128500, "ball out -> corner at 130321")]
    fig, axes = plt.subplots(2 * len(zooms), 1, figsize=(18, 3.6 * len(zooms)),
                             gridspec_kw=dict(height_ratios=[3, 1] * len(zooms)))
    for k, (lo, name) in enumerate(zooms):
        panel(axes[2 * k], axes[2 * k + 1], lo, lo + 180 * FPS, False,
              "frames {}-{}: {}".format(lo, lo + 180 * FPS, name))
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "speed_zoom.png")
    fig.savefig(p2, dpi=100)
    plt.close(fig)
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
