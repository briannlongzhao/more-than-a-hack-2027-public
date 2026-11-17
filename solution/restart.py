"""Dead-ball detection with restarts placed at the restart KICK, not a fixed offset.

v1 ended every stoppage a fixed 2 s before the players sped up. Watching the clips showed the
labels turning H/A while the taker was still standing over a corner or free kick, and the ball
track agrees: where the ball is visible, it rests at the spot and is kicked about 1 s AFTER v1
had already gone live. A set piece is dead until it is taken (Laws of the Game).

  - Stoppage START: as v1 (players stop, shifted for reaction time; ball out of the pitch).
  - Ball RESTING for >= 2 s on a corner, the centre spot or a penalty spot: dead -- a set piece
    waiting to be taken. Joined to a stoppage within 3 s; it ends when the ball leaves the spot.
  - Stoppage END, ball visible: the restart kick -- the first frame near the end of the still
    period where the ball goes from resting (mean speed < REST over the previous REST_WIN) to
    moving fast (> KICK) with a player within NEAR m.
  - Stoppage END, ball not visible (it is tracked on only ~24% of dead frames): the moment the
    players speed up, minus a lag MEASURED on the stoppages where the kick was visible (median of
    still-end minus kick). No ground truth involved -- the ball track calibrates the player signal.
"""

import numpy as np

import features as F
import v1

REST = 2.5           # m/s: the tracked ball jitters ~1-2 m/s even when stationary
REST_WIN = 10        # frames (0.4 s) of rest before the kick
KICK = 6.0           # m/s: clearly struck
NEAR = 2.0           # m: the taker is next to the ball
SEARCH_BEFORE = 4 * F.FPS   # look for the kick from 4 s before the players speed up...
SEARCH_AFTER = 6 * F.FPS    # ...to 6 s after (corners: attackers jostle before the kick)

# A ball RESTING on a restart spot is a set piece waiting to be taken -> dead (Laws of the Game).
# Radii allow for the corner arc (1 m) plus tracking error.
SPOTS = {
    "corner": ([(0, 0), (0, F.WIDTH), (F.LENGTH, 0), (F.LENGTH, F.WIDTH)], 2.0),
    "centre": ([(F.LENGTH / 2, F.WIDTH / 2)], 2.0),
    "penalty": ([(11.0, F.WIDTH / 2), (F.LENGTH - 11.0, F.WIDTH / 2)], 1.5),
}
SPOT_MERGE = 3 * F.FPS      # a ball on a spot within 3 s of a stoppage belongs to it
SPOT_MIN = 2 * F.FPS        # resting that long: a set piece is being set up. Brief slow moments
                            # near a spot happen in live play (checked: <1 s with most players running)


def on_restart_spot(ball, sp, verbose=True):
    """Frames where the tracked ball rests on a corner, the centre spot or a penalty spot."""
    k = np.ones(REST_WIN + 1) / (REST_WIN + 1)
    rest = np.convolve(np.nan_to_num(sp, nan=99.0), k, "same") < REST
    mask = np.zeros(len(ball), bool)
    for name, (points, radius) in SPOTS.items():
        near = np.zeros(len(ball), bool)
        for x, y in points:
            with np.errstate(invalid="ignore"):
                near |= np.hypot(ball[:, 0] - x, ball[:, 1] - y) <= radius
        hit = near & rest
        for s, e in v1.runs(hit):
            if e - s < SPOT_MIN:
                hit[s:e] = False
        if verbose:
            print("  ball resting on a {} spot: {} frames in {} stretches".format(
                name, int(hit.sum()), len(v1.runs(hit))))
        mask |= hit
    return mask


def ball_speed(ball):
    sp = np.full(len(ball), np.nan)
    sp[5:] = np.linalg.norm(ball[5:] - ball[:-5], axis=1) * F.FPS / 5
    return sp


def find_kick(sp, ball, pos, lo, hi):
    for k in range(max(lo, REST_WIN + 5), min(hi, len(sp))):
        if not sp[k] > KICK:
            continue
        before = sp[k - REST_WIN:k]
        if np.isnan(before).any() or before.mean() >= REST:
            continue
        d = np.linalg.norm(pos[k - 5] - ball[k - 5], axis=1)
        if np.nanmin(d) < NEAR:
            return k - 5   # the speed estimate spans 5 frames; the strike is at its start
    return None


def dead_mask(f, share, near, out, graded, verbose=True):
    still = (share < v1.STILL_SHARE) & (near < v1.NEAR_STILL) & graded
    for s, e in v1.runs(~still & graded):
        if e - s < v1.FILL_LIVE and s > 0 and still[s - 1] and e < len(still) and still[e]:
            still[s:e] = True
    stops = [(s, e) for s, e in v1.runs(still) if e - s >= v1.MIN_DEAD]

    sp = ball_speed(f["ball"])
    kicks = [find_kick(sp, f["ball"], f["pos"], max(s, e - SEARCH_BEFORE), e + SEARCH_AFTER)
             for s, e in stops]
    lags = [e - k for (s, e), k in zip(stops, kicks) if k is not None]
    lag = int(np.median(lags)) if lags else v1.LEAD_END
    if verbose:
        print("restart kicks seen in the ball track: {} of {} stoppages; players speed up {:.1f} s"
              " after the kick (median) -> used where the kick is not visible".format(
                  len(lags), len(stops), lag / F.FPS))

    dead = np.zeros_like(still)
    for (s, e), k in zip(stops, kicks):
        start = max(0, s - v1.LEAD_START)
        end = k if k is not None else e - lag
        dead[start:max(start, end)] = True
    out = out & graded
    spot = on_restart_spot(f["ball"], sp, verbose) & graded
    dead |= out | spot
    for s, e in v1.runs(~dead & graded):
        if e - s < v1.OUT_MERGE and ((s > 0 and out[s - 1]) or (e < len(out) and out[e])):
            dead[s:e] = True
        elif e - s < SPOT_MERGE and ((s > 0 and spot[s - 1]) or (e < len(spot) and spot[e])):
            dead[s:e] = True
    return dead & graded
