"""Per-frame arrays from the provided tracking and tagging. Everything downstream starts here.

Reads only the raw files of the data bundle (ATD .txt/.xml, Smart Tagging .xml), so a fresh clone
plus the data reproduces every number. The parsed arrays are cached under data/cache/ (gitignored,
derived from the gated data) purely to save the ~1 min parse on reruns; delete it to rebuild.

All positions are in METRES on a 105.3 x 68.0 pitch, origin at a corner, matching the ATD's
normalised 0-1 axes. Frame index = video frame = submission row.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
HOME, AWAY = 0, 1
CACHE = os.path.join(DATA_DIR, "cache", "features_v1.npz")


def team_of_each_slot(meta_path):
    """0 = home, 1 = away, per track slot. First team in the metadata is home (FIFA/EPTS)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def parse_atd(raw_path, n_slots):
    pos = np.full((N_FRAMES, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1  # ATD frame_id counts from 1
            if not 0 <= i < N_FRAMES:
                continue
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                x, y, s = group.split(",")[:3]
                pos[i, slot] = (float(x) * LENGTH, float(y) * WIDTH)
                if s[0] != "N":
                    speed[i, slot] = float(s)
            if parts[2][0] != "N":
                x, y = parts[2].split(",")[:2]
                ball[i] = (float(x) * LENGTH, float(y) * WIDTH)
    return pos, speed, ball


def graded_mask():
    """True inside the two halves. Outside them nothing is scored and we write X."""
    mask = np.zeros(N_FRAMES, bool)
    for _, lo, hi in half_video_blocks(GAME):
        mask[lo:hi + 1] = True
    return mask


def set_pieces():
    """Smart Tagging SET PIECES as dicts with start/end frame, type and team (or None for N/A)."""
    out = []
    for e in load_smart_tagging.load():
        if e["code"].upper() != "SET PIECES":
            continue
        team = e["tags"].get("Team", "N/A")
        out.append({"start": e["start_frame"], "end": e["end_frame"],
                    "type": e["tags"].get("Type", ""),
                    "team": None if team.upper() == "N/A" else team})
    return out


def load(rebuild=False):
    """dict of per-frame arrays: pos (n,46,2) m, speed (n,46) m/s, ball (n,2) m, team (46,)."""
    paths = asset_paths()
    team = team_of_each_slot(paths["atd_meta"])
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE)
        return {"pos": z["pos"], "speed": z["speed"], "ball": z["ball"], "team": team}
    pos, speed, ball = parse_atd(paths["atd_raw"], len(team))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, pos=pos, speed=speed, ball=ball)
    return {"pos": pos, "speed": speed, "ball": ball, "team": team}


def home_team_name():
    root = ET.parse(asset_paths()["atd_meta"]).getroot()
    return [t.findtext("Name") for t in root.iter("Team")][0]
