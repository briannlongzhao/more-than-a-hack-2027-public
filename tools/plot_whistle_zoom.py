"""Zoomed spectrograms (+/-3 s) around public-GT dead-ball starts, 2-6 kHz, to see whistle tones."""
import os, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import features as F
SR = 16000
gt = np.array([l.split(",")[1].strip() for l in open(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv")).readlines()[1:]])
starts = list(np.flatnonzero((gt[1:] == "D") & (gt[:-1] != "D")) + 1)
fig, axes = plt.subplots(1, len(starts), figsize=(5 * len(starts), 4.2))
for ax, s in zip(axes, starts):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(s / F.FPS - 3), "-t", "6", "-i", F.asset_paths()["video"],
                                   "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"])
    x = np.frombuffer(raw, np.int16).astype(np.float32)
    n, hop = 2048, 160
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    S = 10 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-9)
    f = np.fft.rfftfreq(n, 1 / SR); m = (f >= 2000) & (f <= 6000)
    ax.imshow(S[:, m].T, aspect="auto", origin="lower", extent=[-3, 3, 2000, 6000], cmap="magma",
              vmin=np.percentile(S[:, m], 20), vmax=np.percentile(S[:, m], 99.95))
    ax.axvline(0, color="cyan", lw=1)
    ax.set_title("GT D starts at frame %d" % s, fontsize=10); ax.set_xlabel("s relative to GT D start")
axes[0].set_ylabel("Hz")
fig.tight_layout(); p = os.path.join(F.DATA_DIR, "artifact", "whistle_zoom.png"); fig.savefig(p, dpi=90); print(p)
