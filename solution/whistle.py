"""Referee whistle detection from the tactical video's audio track.

The track carries quiet stadium sound (about -59 dB mean). Around stoppages in the public window a
referee whistle shows up as a FLAT, prominent tone near 3.8 kHz, usually with a companion near
4.3 kHz, lasting 0.2-1.5 s (tools/plot_whistle_zoom.py). Crowd noise in the same range sweeps in
pitch, so the detector requires a STEADY pitch, not just energy.

Per video frame (40 ms hop, 128 ms window, 16 kHz mono):
  - the strongest bin in 3.55-4.05 kHz, and its prominence over the median of 3.0-4.6 kHz
  - a frame is whistle-like when that prominence is >= PROM dB and the peak frequency moved less
    than STEADY_HZ since the previous frame
  - runs of whistle-like frames >= MIN_LEN (gaps <= GAP merged) are whistle events

Thresholds come from what a whistle physically looks like in the spectrogram, not from fitting the
public labels. Cached to data/cache/whistles.npz (event onset/offset frames, per-frame score).

    python solution/whistle.py
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import v1  # noqa: E402

CACHE = os.path.join(F.DATA_DIR, "cache", "whistles.npz")
SR = 16000
HOP = SR // F.FPS            # 640 samples = one video frame
NFFT = 2048                  # 128 ms window, 7.8 Hz bins
BAND = (3550.0, 4050.0)      # whistle fundamental
REF = (3000.0, 4600.0)       # local reference for prominence
PROM = 15.0                  # dB above the local median (candidate frames)
STRICT_PROM = 28.0           # event peak: a referee whistle is the loudest tone in the stadium
STRICT_SHARP = 10.0          # event median: a tone is >= 10 dB above the bins 80-200 Hz away;
                             # crowd noise in this band is a broad slope, not a narrow line
STEADY_HZ = 40.0             # max peak-frequency change between consecutive frames
MIN_LEN = 5                  # frames (0.2 s)
GAP = 3                      # frames merged inside one whistle


def decode_audio():
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", F.asset_paths()["video"], "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "s16le", "-"])
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def frame_scores(x):
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    ref = (freqs >= REF[0]) & (freqs <= REF[1])
    win = np.hanning(NFFT).astype(np.float32)
    n = min(F.N_FRAMES, (len(x) - NFFT) // HOP)
    prom = np.zeros(F.N_FRAMES, np.float32)
    peak_hz = np.zeros(F.N_FRAMES, np.float32)
    sharp = np.zeros(F.N_FRAMES, np.float32)
    comp = np.zeros(F.N_FRAMES, np.float32)
    df = freqs[1] - freqs[0]
    band_idx = np.flatnonzero(band)
    near_off = np.r_[-np.arange(int(80 / df), int(200 / df) + 1), np.arange(int(80 / df), int(200 / df) + 1)]
    comp_off = np.arange(int(400 / df), int(600 / df) + 1)
    for lo in range(0, n, 4096):
        hi = min(n, lo + 4096)
        idx = (np.arange(lo, hi) * HOP)[:, None] + np.arange(NFFT)[None, :]
        # centre the window on the video frame
        idx = np.clip(idx - NFFT // 2 + HOP // 2, 0, len(x) - 1)
        S = 10 * np.log10(np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2 + 1e-12)
        rows = np.arange(len(S))
        k = band_idx[np.argmax(S[:, band], axis=1)]
        med = np.median(S[:, ref], axis=1)
        pk = S[rows, k]
        prom[lo:hi] = pk - med
        peak_hz[lo:hi] = freqs[k]
        # sharpness: the peak against the loudest bin 80-200 Hz either side (a tone is narrow)
        sharp[lo:hi] = pk - S[rows[:, None], k[:, None] + near_off[None, :]].max(axis=1)
        # companion: strongest bin 400-600 Hz above the peak, against the same local median
        comp[lo:hi] = S[rows[:, None], k[:, None] + comp_off[None, :]].max(axis=1) - med
    return prom, peak_hz, sharp, comp


def detect(prom, peak_hz):
    steady = np.r_[False, np.abs(np.diff(peak_hz)) <= STEADY_HZ]
    hit = (prom >= PROM) & steady
    for s, e in v1.runs(~hit):
        if e - s <= GAP and s > 0 and e < len(hit) and hit[s - 1] and hit[e]:
            hit[s:e] = True
    events = [(s, e) for s, e in v1.runs(hit) if e - s >= MIN_LEN]
    return np.array(events, dtype=np.int64).reshape(-1, 2)


def strict_onsets(rebuild=False):
    """Onset frames of whistle events that are loud and narrow enough to be the referee's."""
    events, _ = load(rebuild)
    z = np.load(CACHE)
    keep = [s for s, e in events
            if z["prom"][s:e].max() >= STRICT_PROM and np.median(z["sharp"][s:e]) >= STRICT_SHARP]
    return np.array(keep, dtype=np.int64)


def load(rebuild=False):
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE)
        return z["events"], z["prom"]
    prom, peak_hz, sharp, comp = frame_scores(decode_audio())
    events = detect(prom, peak_hz)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, events=events, prom=prom, peak_hz=peak_hz, sharp=sharp, comp=comp)
    return events, prom


if __name__ == "__main__":
    ev, _ = load(rebuild=True)
    graded = F.graded_mask()
    print("whistle events: {} ({} in graded play), median length {:.2f} s".format(
        len(ev), int(graded[ev[:, 0]].sum()) if len(ev) else 0,
        np.median(ev[:, 1] - ev[:, 0]) / F.FPS if len(ev) else 0))
