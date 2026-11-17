"""v10: v9 + throw-ins restart when the ball re-enters the pitch.

A throw-in is dead until the ball, released by the thrower, enters the field of play (Laws of the
Game). While the thrower holds it, the tracked ball is usually OUTSIDE the touchline; the frame it
crosses back inside is the restart. v9 found that moment on time for half of the throw-ins and
fell back to rest->kick or the players' speed for the rest (0.8-7 s late in 7 cases).

Rule: for every stoppage, if the ball is tracked beyond a touchline (y < -OUT or y > 68 + OUT),
comes back inside (OUT < y < 68 - OUT) within ENTRY s, and then leaves the touchline area FAST
(the throw, see throws()), with that moment between SEARCH_BEFORE s before and SEARCH_AFTER s
after the stoppage's current end, the stoppage ends at the throw. A ball merely drifting back over
the line (carried or rolled to the thrower) does not count.

    python solution/v10.py                # -> evaluation/results/v10.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import v1  # noqa: E402
import v5  # noqa: E402
import v8  # noqa: E402
import v9  # noqa: E402

OUT = 0.5                      # m beyond / inside the touchline
ENTRY = 1.0                    # s: outside -> inside within this = the throw
SEARCH_BEFORE = 4 * F.FPS      # the throw may come before the current end...
SEARCH_AFTER = 6 * F.FPS       # ...or after it
THROW_SPEED = 5.0              # m/s: a thrown ball, not one carried or rolled back
LINE_ZONE = 3.0                # m: the throw starts at the touchline
THROW_WAIT = 4 * F.FPS         # the throw follows the ball's return to the line within 4 s


def reentries(ball):
    """Frames where a ball tracked beyond a touchline is next seen inside the pitch within ENTRY s."""
    y = ball[:, 1]
    with np.errstate(invalid="ignore"):
        outside = (y < -OUT) | (y > F.WIDTH + OUT)
        inside = (y > OUT) & (y < F.WIDTH - OUT)
    win = int(ENTRY * F.FPS)
    frames = []
    for k in np.flatnonzero(outside[:-1] & ~outside[1:]):     # last frame seen outside
        nxt = np.flatnonzero(inside[k + 1:k + 1 + win])
        if len(nxt):
            frames.append(k + 1 + nxt[0])
    return np.array(frames, dtype=np.int64)


def throws(ball):
    """The throw itself: after a re-entry, the first moment the ball leaves the touchline area FAST.

    Crossing the line is not enough -- a ball carried or rolled back to the thrower drifts over it
    slowly with nobody near (seen in a clip: 5 s of D called live before the actual throw). A thrown
    ball moves faster than THROW_SPEED while still within LINE_ZONE m of the touchline.
    """
    sp = np.full(len(ball), np.nan)
    sp[5:] = np.linalg.norm(ball[5:] - ball[:-5], axis=1) * F.FPS / 5
    y = ball[:, 1]
    with np.errstate(invalid="ignore"):
        at_line = (y < LINE_ZONE) | (y > F.WIDTH - LINE_ZONE)
        fast = sp > THROW_SPEED
    out = []
    for r in reentries(ball):
        hit = np.flatnonzero(fast[r:r + THROW_WAIT] & at_line[r - 5:r + THROW_WAIT - 5])
        if len(hit):
            out.append(r + hit[0] - 5)   # speed spans 5 frames: the release is at its start
    return np.array(out, dtype=np.int64)


def dead_mask(f, share, near, out, graded):
    dead = v9.dead_mask(f, share, near, out, graded)
    entries = throws(f["ball"])
    moved_early, moved_late = 0, 0
    for s, e in v1.runs(dead):
        cand = entries[(entries > s) & (entries >= e - SEARCH_BEFORE) & (entries <= e + SEARCH_AFTER)]
        if not len(cand):
            continue
        r = cand[0]
        if r < e:
            dead[r:e] = False
            moved_early += 1
        elif r > e:
            dead[e:r] = True
            moved_late += 1
    print("throw-in restarts set at the ball re-entering the pitch: {} earlier, {} later".format(
        moved_early, moved_late))
    return dead & graded


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v10.csv"))
    args = p.parse_args()
    labels = v5.run(dead_fn=dead_mask, restart_touch=v8.RESTART_TOUCH)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    v1.summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
