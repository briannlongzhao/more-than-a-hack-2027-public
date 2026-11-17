"""v4a / v4b: quick experiments putting PathCRF's team into v3's output.

Both keep v3's labels for dead ball. They differ only in which live frames take PathCRF's team:
  v4a  every in-play frame where PathCRF has an answer
  v4b  only in-play frames where the ball is NOT tracked (PathCRF fills the ball gaps)

v4a showed PathCRF names the team better than v3 but fragments possessions; that led to v5.

    python solution/v4.py                 # -> evaluation/results/v4a_pathcrf_all.csv, v4b_pathcrf_gaps.csv
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
from pathcrf_adapter import OUT as PATHCRF_CACHE  # noqa: E402

RESULTS = os.path.join(F.ROOT, "evaluation", "results")


def read(name):
    with open(os.path.join(RESULTS, name + ".csv")) as handle:
        next(handle)
        return np.array([line.strip().split(",")[1] for line in handle], dtype="<U1")


def write(name, labels):
    with open(os.path.join(RESULTS, name + ".csv"), "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("->", os.path.join("evaluation", "results", name + ".csv"))


if __name__ == "__main__":
    v3 = read("v3")
    pc = np.load(PATHCRF_CACHE)
    ball_seen = ~np.isnan(F.load()["ball"][:, 0])
    live = np.isin(v3, ["H", "A"])

    a = v3.copy()
    m = live & (pc >= 0)
    a[m] = np.where(pc[m] == 0, "H", "A")
    write("v4a_pathcrf_all", a)

    b = v3.copy()
    m = live & (pc >= 0) & ~ball_seen
    b[m] = np.where(pc[m] == 0, "H", "A")
    write("v4b_pathcrf_gaps", b)
