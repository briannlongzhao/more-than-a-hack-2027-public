"""Exploration: is there usable stadium sound (referee whistle) in the tactical video's audio?

Decodes the public 5-minute window to 16 kHz mono, plots a spectrogram (normalised, since the
track is very quiet), the energy in the whistle band (2.5-4.5 kHz) relative to the rest, and the
public GT so dead-ball starts can be compared with whistle-like peaks. Writes
data/artifact/audio_check.png.
"""
import os, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import features as F

LO, HI = 11185, 18685
SR = 16000
raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(LO / F.FPS), "-t", str((HI - LO) / F.FPS),
                               "-i", F.asset_paths()["video"], "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"])
x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
print("samples %d, rms %.2e, peak %.2e, fraction of exact zeros %.1f%%" % (len(x), np.sqrt(np.mean(x**2)), np.abs(x).max(), 100 * np.mean(x == 0)))
n, hop = 1024, 640                     # 640 samples = 1 video frame at 16 kHz
win = np.hanning(n)
frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * win
S = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-20
freqs = np.fft.rfftfreq(n, 1 / SR)
band = (freqs >= 2500) & (freqs <= 4500)
other = (freqs >= 300) & (freqs < 2500)
ratio = 10 * np.log10(S[:, band].max(axis=1) / np.median(S[:, other], axis=1))
gt = np.array([l.split(",")[1].strip() for l in open(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv")).readlines()[1:]])[LO:HI]
t = np.arange(len(S)) * hop / SR
fig, ax = plt.subplots(3, 1, figsize=(18, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 2, 0.4]))
ax[0].imshow(10 * np.log10(S.T), aspect="auto", origin="lower", extent=[0, t[-1], 0, SR / 2], cmap="magma",
             vmin=np.percentile(10 * np.log10(S), 5), vmax=np.percentile(10 * np.log10(S), 99.9))
ax[0].set_ylabel("Hz"); ax[0].set_title("spectrogram, public window (normalised colour scale)", loc="left", fontsize=10)
ax[1].plot(t, ratio, lw=0.6, color="k"); ax[1].set_ylabel("whistle band peak vs\nmedian 0.3-2.5 kHz (dB)")
cols = {"H": "#2b6eff", "A": "#eb2832", "D": "#999999"}
tg = np.arange(len(gt)) / F.FPS
for lab, c in cols.items():
    ax[2].fill_between(tg, 0, (gt == lab).astype(float), step="mid", color=c, lw=0)
ax[2].set_yticks([]); ax[2].set_xlabel("seconds from frame %d (public GT strip: grey = D)" % LO)
for s in np.flatnonzero((gt[1:] == "D") & (gt[:-1] != "D")) + 1:
    for a in ax[:2]:
        a.axvline(s / F.FPS, color="cyan", lw=1, alpha=0.8)
fig.tight_layout()
p = os.path.join(F.DATA_DIR, "artifact", "audio_check.png"); fig.savefig(p, dpi=90); print(p)
