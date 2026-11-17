"""Run the pretrained PathCRF model (third_party/pathcrf, KDD 2026) on our tracking.

PathCRF infers who has the ball from PLAYER trajectories only, so it can speak during the 41% of
frames where our ball is not tracked. It was trained on 7 public Bundesliga matches (Sportec/DFL),
not on this game -- allowed by RULES.md (pretrained public model, labels from other games).

What has to be bridged, and how:
  - Our 46 slots are tracklets, not players. Inside each in-play episode, tracklet pieces of a team
    are STITCHED into at most 11 chains (a piece continues the chain whose last position it can
    reach in time), and gaps inside a chain are linearly interpolated, ends held. PathCRF skips any
    episode with a NaN, so every chain must be complete.
  - Episodes = in-play stretches from our own dead-ball detector (v1). PathCRF has no dead state.
  - Kinematics are recomputed exactly as PathCRF's preprocessing does (Savitzky-Golay, 25 fps).
  - Our pitch is 105.3 m long; PathCRF assumes 105 m, so x is rescaled.
Output per frame: the team of the possession edge's source player (the passer while the ball
travels -- possession stays with the last team, CHALLENGE.md), or of the destination when the
source is an outside node (a restart).

Needs the `pathcrf` env (torch, torch_geometric). Writes data/cache/pathcrf_team.npy:
-1 = no output, 0 = home, 1 = away, per video frame.

    ~/miniconda3/envs/pathcrf/bin/python solution/pathcrf_adapter.py [--max-episodes N]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PCRF = os.path.join(ROOT, "third_party", "pathcrf")
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import v1  # noqa: E402

OUT = os.path.join(F.DATA_DIR, "cache", "pathcrf_team.npy")
EDGES = os.path.join(F.DATA_DIR, "cache", "pathcrf_edges.npz")
TRIAL = 140                     # Dynamic Masked CRF, the paper's full model
MIN_EPISODE = 2 * F.FPS         # shorter in-play stretches are skipped
LINK_SPEED = 8.0                # m/s: max implied speed to continue a chain across a gap
LINK_SLACK = 3.0                # m: plus this, for tracking noise
X_SCALE = 105.0 / F.LENGTH


def pieces_of(slot_xy):
    """Contiguous non-NaN runs of one slot: list of (start, end)."""
    return v1.runs(~np.isnan(slot_xy[:, 0]))


def stitch_team(pos_ep, slots, max_chains=11):
    """Chain tracklet pieces of one team inside an episode. Returns (n_chains, T, 2), gap-free."""
    T = pos_ep.shape[0]
    pcs = sorted((s, e, k) for k in slots for s, e in pieces_of(pos_ep[:, k]))
    chains = []  # each: list of (start, end, slot)
    for s, e, k in pcs:
        best, best_cost = None, None
        for c in chains:
            ls, le, lk = c[-1]
            if le > s:              # overlaps in time: a different person
                continue
            gap = (s - le + 1) / F.FPS
            dist = np.linalg.norm(pos_ep[s, k] - pos_ep[le - 1, lk])
            if dist <= LINK_SLACK + LINK_SPEED * gap and (best_cost is None or dist < best_cost):
                best, best_cost = c, dist
        if best is None:
            chains.append([(s, e, k)])
        else:
            best.append((s, e, k))
    # keep the chains that cover the most frames (duplicates/false tracks are short)
    chains.sort(key=lambda c: -sum(e - s for s, e, _ in c))
    out = np.full((min(len(chains), max_chains), T, 2), np.nan, dtype=np.float64)
    for i, c in enumerate(chains[:max_chains]):
        for s, e, k in c:
            out[i, s:e] = pos_ep[s:e, k]
        t = np.arange(T)
        ok = ~np.isnan(out[i, :, 0])
        for d in range(2):
            out[i, :, d] = np.interp(t, t[ok], out[i, ok, d])  # interior gaps linear, ends held
    return out


def kinematics(xy):
    """vx, vy, speed, accel the way PathCRF's match_data.py computes them."""
    def sg(v, w):
        w = min(w, len(v) if len(v) % 2 else len(v) - 1)
        return savgol_filter(v, window_length=w, polyorder=2) if w > 2 else v
    vx = sg(np.diff(xy[:, 0], prepend=xy[0, 0]) * F.FPS, 15)
    vy = sg(np.diff(xy[:, 1], prepend=xy[0, 1]) * F.FPS, 15)
    speed = np.hypot(vx, vy)
    accel = sg(np.diff(speed, prepend=speed[0]) * F.FPS, 9)
    return vx, vy, speed, accel


def build_tracking(f, live, max_episodes=None):
    """Wide DataFrame in PathCRF's format, one row per in-play frame."""
    pos, team = f["pos"], f["team"]
    halves = {fr: h for h, lo, hi in F.half_video_blocks(F.GAME) for fr in (lo, hi)}
    episodes = [(s, e) for s, e in v1.runs(live) if e - s >= MIN_EPISODE]
    if max_episodes:
        episodes = episodes[:max_episodes]
    frames = []
    for ep_id, (s, e) in enumerate(episodes, start=1):
        half = 1 if s < F.half_video_blocks(F.GAME)[1][1] else 2
        cols = {"frame_id": np.arange(s, e), "period_id": half, "phase_id": half,
                "episode_id": ep_id}
        for side, name in ((F.HOME, "home"), (F.AWAY, "away")):
            chains = stitch_team(pos[s:e], np.flatnonzero(team == side))
            for i, xy in enumerate(chains):
                xy = xy.copy()
                xy[:, 0] *= X_SCALE
                vx, vy, sp, ac = kinematics(xy)
                p = "{}_{}".format(name, i + 1)
                cols.update({p + "_x": xy[:, 0], p + "_y": xy[:, 1], p + "_vx": vx,
                             p + "_vy": vy, p + "_speed": sp, p + "_accel": ac})
        frames.append(pd.DataFrame(cols))
    del halves
    return pd.concat(frames, ignore_index=True), episodes


def edge_team(src, dst):
    def team_of(node):
        if isinstance(node, str) and node.startswith("home_"):
            return F.HOME
        if isinstance(node, str) and node.startswith("away_"):
            return F.AWAY
        return -1
    t = team_of(src)
    return t if t >= 0 else team_of(dst)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-episodes", type=int, default=None, help="quick test on the first N")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    f = F.load()
    graded = F.graded_mask()
    share, near, _, out = v1.signals(f)
    live = graded & ~v1.dead_mask(share, near, out, graded)
    tracking, episodes = build_tracking(f, live, args.max_episodes)
    print("episodes {}, frames {}, player columns {}".format(
        len(episodes), len(tracking), sum(c.endswith("_x") for c in tracking.columns)))

    sys.path.insert(0, PCRF)
    cwd = os.getcwd()
    os.chdir(PCRF)  # the repo resolves saved/ relative to itself
    import torch
    from inference import inference
    from models.utils import build_model, load_trial_args, resolve_model_path
    save_path = "saved/{:03d}".format(TRIAL)
    trial_args = load_trial_args(save_path)
    model = build_model(trial_args, device=args.device)
    state = torch.load(resolve_model_path(save_path, "state_dict_best_acc.pt"),
                       map_location=args.device, weights_only=False)
    model.load_state_dict(state)
    os.chdir(cwd)

    with torch.no_grad():
        _, _, micro, stats = inference(model, tracking, use_crf=True, decode="indep",
                                       window_seconds=trial_args.get("window_seconds"),
                                       evaluate=False, correct_episode_lasts=False)
    print("inference time {:.0f} s".format(stats["inference_time"]))

    result = np.full(F.N_FRAMES, -1, dtype=np.int8)
    if {"edge_src", "edge_dst"}.issubset(micro.columns):
        m = micro.dropna(subset=["edge_src"])
        teams = np.array([edge_team(s, d) for s, d in zip(m["edge_src"], m["edge_dst"])])
        result[m.index.to_numpy(dtype=int)] = teams
        # the raw possession edges too: edges to/from out_* nodes are PathCRF's out-of-play events
        np.savez(EDGES, frame=m.index.to_numpy(dtype=np.int64),
                 src=np.asarray(m["edge_src"].astype(str).tolist(), dtype=str),
                 dst=np.asarray(m["edge_dst"].astype(str).tolist(), dtype=str))
    covered = np.sum(result >= 0)
    print("frames with a PathCRF team: {} ({:.0%} of in-play frames used)".format(
        covered, covered / max(1, len(tracking))))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, result)
    print("->", OUT)


if __name__ == "__main__":
    main()
