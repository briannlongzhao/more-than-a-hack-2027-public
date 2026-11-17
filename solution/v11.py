"""v11: v10 + after a goal the ball is dead until the kick-off.

Celebrations look like play to a speed detector: up to two thirds of the players run for 10-30 s
after a goal (seen in a clip), and players jog back into position before the kick-off. v10 called
17-61 s "live" between each of the 4 goals and its kick-off. By the Laws of the Game the ball is
dead from the goal until the kick-off is taken.

Uses the provided Smart Tagging (automatic, not ground truth):
  - GOAL MOMENT: the last frame the ball is tracked inside a penalty box within GOAL_WINDOW s of the
    GOALS tag's start (the shot); if the ball is not tracked there, the tag start.
  - KICK-OFF: the ball leaving the centre spot fast near the following "Kick Off (After Goal)" tag;
    if not visible, the tag start + KICKOFF_DELAY (measured: on the 3 kick-offs visible in the ball
    track the ball moved 3.2-3.5 s after the tag start).
Everything between is D.

    python solution/v11.py                # -> evaluation/results/v11.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import load_smart_tagging as ST  # noqa: E402
import restart  # noqa: E402
import v1  # noqa: E402
import v5  # noqa: E402
import v8  # noqa: E402
import v10  # noqa: E402

GOAL_WINDOW = 10 * F.FPS        # the shot is within 10 s of the GOALS tag start
KICKOFF_DELAY = int(3.5 * F.FPS)
KICKOFF_SEARCH = (-5 * F.FPS, 10 * F.FPS)
CENTRE_R = 3.0                  # m
KICK_SPEED = 5.0                # m/s
BOX_DEPTH, BOX_HALF = 16.5, 20.16


def goal_intervals(ball):
    events = ST.load()
    goals = [e["start_frame"] for e in events if e["code"].upper() == "GOALS"]
    kickoffs = [e["start_frame"] for e in events if e["tags"].get("Type") == "Kick Off (After Goal)"]
    sp = restart.ball_speed(ball)
    with np.errstate(invalid="ignore"):
        in_box = ((ball[:, 0] < BOX_DEPTH) | (ball[:, 0] > F.LENGTH - BOX_DEPTH)) & \
                 (np.abs(ball[:, 1] - F.WIDTH / 2) < BOX_HALF)
        at_centre = np.hypot(ball[:, 0] - F.LENGTH / 2, ball[:, 1] - F.WIDTH / 2) < CENTRE_R
    out = []
    for gs in goals:
        after = [k for k in kickoffs if k > gs]
        if not after:
            continue
        ko = after[0]
        seen = np.flatnonzero(in_box[gs - GOAL_WINDOW:gs + GOAL_WINDOW])
        goal = gs - GOAL_WINDOW + seen[-1] if len(seen) else gs
        lo, hi = ko + KICKOFF_SEARCH[0], ko + KICKOFF_SEARCH[1]
        kick = next((k - 5 for k in range(lo, hi) if at_centre[k - 5] and sp[k] > KICK_SPEED), None)
        end = kick if kick is not None else ko + KICKOFF_DELAY
        out.append((int(goal), int(end), kick is not None))
    return out


def dead_mask(f, share, near, out, graded):
    dead = v10.dead_mask(f, share, near, out, graded)
    for goal, end, seen in goal_intervals(f["ball"]):
        added = int(np.sum(~dead[goal:end] & graded[goal:end]))
        dead[goal:end] = True
        # the kick-off ends the stoppage: anything v10 still had dead right after it is released
        # only if the ball was seen kicked off (otherwise keep v10's own restart evidence)
        if seen:
            nxt = end
            while nxt < len(dead) and dead[nxt] and nxt < end + 2 * F.FPS:
                nxt += 1
            if nxt < end + 2 * F.FPS:
                dead[end:nxt] = False
        print("goal at frame {}: dead until kick-off at {} ({}), {:.0f} s of 'live' removed".format(
            goal, end, "seen in ball track" if seen else "tag + 3.5 s", added / F.FPS))
    return dead & graded


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(F.ROOT, "evaluation", "results", "v11.csv"))
    args = p.parse_args()
    labels = v5.run(dead_fn=dead_mask, restart_touch=v8.RESTART_TOUCH)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    v1.summary(labels)
    print("->", os.path.relpath(args.out, F.ROOT))
