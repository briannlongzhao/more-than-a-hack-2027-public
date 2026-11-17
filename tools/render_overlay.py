"""Render the tactical video with the automatic tracking drawn on top, for inspecting the data.

What is drawn on every frame
----------------------------
  players     dot at each tracked slot's projected position, coloured by team, labelled with
              the slot number (a tracklet index, NOT a shirt number)
  ball        yellow dot + 1 s trail when tracked. When the ball is MISSING: red banner with the
              gap's elapsed / total length, and a hollow grey marker at the last known position
  nearest     line from the ball to the nearest tracked player, with distance in metres
              (what the reference baseline uses)
  HUD         frame index (= submission row), half and match clock, graded or not, player counts
              per team, homography reprojection error, baseline label, public GT label (first
              5 minutes only), active Smart Tagging phases
  minimap     top-down pitch with every slot and the ball
  timeline    +/-30 s strips: ball tracked/missing, public GT, baseline(+/-2 s); cursor = now

Output keeps every frame of the source, so frame N of the output is frame N of the tactical video
and row N of a submission.

    python tools/render_overlay.py                       # whole match, parallel chunks
    python tools/render_overlay.py --start 11185 --frames 7500 --name public_window
    python tools/render_overlay.py --png 15000           # one annotated still, for a quick look

The output lands in data/artifact/, which is gitignored: an overlay video is derived from the
match video and must not be published (see RULES.md).
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAMES, GAME, asset_paths, half_video_blocks  # noqa: E402
import baseline  # noqa: E402
import load_homography  # noqa: E402
import load_smart_tagging  # noqa: E402

OUT_DIR = os.path.join(DATA_DIR, "artifact")
CACHE = os.path.join(OUT_DIR, "cache", "overlay_arrays.npz")
# Optional: a submission to show next to the baseline, and the touches it was built from.
EXTRA = os.path.join(OUT_DIR, "cache", "overlay_extra.npz")
TOUCH_SHOW = 8  # frames either side of a touch during which it is drawn
W_IMG, H_IMG = 1920, 1080
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]

# Colours (RGB). Home = Barcelona, Away = Athletic.
C_HOME = (40, 110, 255)
C_AWAY = (235, 40, 50)
C_BALL = (255, 230, 0)
C_DEAD = (150, 150, 150)
C_NONE = (35, 35, 35)
C_OK = (40, 200, 90)
C_MISS = (230, 40, 40)
LABEL_RGB = {"H": C_HOME, "A": C_AWAY, "D": C_DEAD, "X": C_NONE}

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(size, bold=False):
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default(size)


# ---------------------------------------------------------------------------------------------
# Precomputation: everything per-frame, once, cached to disk so chunk workers just load arrays.
# ---------------------------------------------------------------------------------------------

def parse_atd(raw_path, n_slots):
    players = np.full((N_FRAMES, n_slots, 2), np.nan, dtype=np.float32)
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
                if group[0] != "N":
                    x, y = group.split(",")[:2]
                    players[i, slot] = (float(x), float(y))
            if parts[2][0] != "N":
                x, y = parts[2].split(",")[:2]
                ball[i] = (float(x), float(y))
    return players, ball


def precompute():
    if os.path.exists(CACHE):
        return dict(np.load(CACHE))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    paths = asset_paths()
    team = baseline.team_of_each_track(paths["atd_meta"])
    print("parsing tracking (about a minute) ...", flush=True)
    players, ball = parse_atd(paths["atd_raw"], len(team))

    # Nearest tracked player to the ball, in metres -- the baseline's step 1.
    scale = np.array([LENGTH, WIDTH], dtype=np.float32)
    d = np.linalg.norm((players - ball[:, None, :]) * scale, axis=2)
    has = ~np.isnan(d).all(axis=1)
    nearest = np.full(N_FRAMES, -1, dtype=np.int16)
    nearest[has] = np.nanargmin(d[has], axis=1)
    nearest_d = np.full(N_FRAMES, np.nan, dtype=np.float32)
    nearest_d[has] = d[has, nearest[has]]
    codes = np.where(nearest >= 0, team[np.clip(nearest, 0, None)], baseline.NOBODY).astype(np.int8)
    base50 = baseline.smooth(codes, 50)

    # Ball gaps: elapsed and total length of the current gap, and the last known position.
    missing = np.isnan(ball[:, 0])
    gap_elapsed = np.zeros(N_FRAMES, dtype=np.int32)
    gap_total = np.zeros(N_FRAMES, dtype=np.int32)
    last_ball = ball.copy()
    run = 0
    for i in range(N_FRAMES):
        if missing[i]:
            run += 1
            gap_elapsed[i] = run
            if i > 0:
                last_ball[i] = last_ball[i - 1]
        else:
            run = 0
    i = N_FRAMES - 1
    while i >= 0:
        if missing[i]:
            j = i
            while j >= 0 and missing[j]:
                j -= 1
            gap_total[j + 1:i + 1] = i - j
            i = j
        else:
            i -= 1

    H, reproj, valid = load_homography.load()

    gt = np.full(N_FRAMES, "X", dtype="<U1")
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    with open(gt_path) as handle:
        next(handle)
        for line in handle:
            f, lab = line.strip().split(",")
            gt[int(f)] = lab.strip().upper()

    home, away = baseline.team_names(paths["atd_meta"])
    arrays = dict(players=players, ball=ball, team=team, nearest=nearest, nearest_d=nearest_d,
                  base50=base50, gap_elapsed=gap_elapsed, gap_total=gap_total,
                  last_ball=last_ball, H=H.astype(np.float32), reproj=reproj.astype(np.float32),
                  valid=valid, gt=gt, names=np.array([home, away]))
    np.savez(CACHE, **arrays)
    print("cached ->", os.path.relpath(CACHE, ROOT), flush=True)
    return arrays


def precompute_extra(labels_path):
    """Labels of a submission CSV + v3's touch events, cached for the chunk workers."""
    sys.path.insert(0, os.path.join(ROOT, "solution"))
    import features as SF
    import v2 as SV2
    import v3 as SV3
    with open(labels_path) as handle:
        next(handle)
        pred = np.array([line.strip().split(",")[1].upper() for line in handle], dtype="<U1")
    f = SF.load()
    near_team, dist = SV2.nearest_player(f)
    tf, tt, _ = SV2.detect_touches(f, near_team, dist)
    contested = (SV3.team_min_dist(f, tf) <= SV3.CONTEST_NEAR).all(axis=1)
    name = os.path.splitext(os.path.basename(labels_path))[0]
    np.savez(EXTRA, pred=pred, touch_frames=tf, touch_teams=tt, contested=contested,
             pred_name=np.array(name))


def strip_colors(labels):
    lut = np.array([LABEL_RGB[k] for k in "HADX"], dtype=np.uint8)
    idx = np.searchsorted(np.array(list("ADHX")), labels)  # sorted order A, D, H, X
    remap = np.array([1, 2, 0, 3])  # A->1, D->2, H->0, X->3 in "HADX"
    return lut[remap[idx]]


# ---------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------

class Renderer:
    MM_W, MM_H = 420, int(420 * 68.0 / 105.3)
    MM_X0, MM_Y0 = W_IMG - 420 - 20, H_IMG - int(420 * 68.0 / 105.3) - 150
    TL_ROW, TL_HALF = 22, 750  # timeline: +/-30 s

    def __init__(self, a, events, extra=None):
        self.a = a
        self.events = events
        self.extra = extra
        self.TL_Y0 = H_IMG - (146 if extra is not None else 118)
        # Clean mode (for the comparison site): no burned-in label rows or timeline strips --
        # the web page draws those, for every version, synced to the video.
        self.clean = os.environ.get("OVERLAY_CLEAN") == "1"
        self.f_small = font(18)
        self.f_med = font(22)
        self.f_big = font(34, bold=True)
        self.f_slot = font(14, bold=True)
        self.halves = half_video_blocks(GAME)
        graded = np.zeros(N_FRAMES, bool)
        for _, lo, hi in self.halves:
            graded[lo:hi + 1] = True
        self.graded = graded
        self.ball_rgb = np.where(np.isnan(a["ball"][:, :1]), np.array([C_MISS]),
                                 np.array([C_OK])).astype(np.uint8)
        self.ball_rgb[~graded] = (np.array(self.ball_rgb[~graded]) * 0.35).astype(np.uint8)
        self.gt_rgb = strip_colors(a["gt"])
        self.base_rgb = strip_colors(a["base50"])
        if extra is not None:
            self.pred = extra["pred"]
            self.pred_name = str(extra["pred_name"])
            self.pred_rgb = strip_colors(self.pred)
            self.tf, self.tt, self.tc = extra["touch_frames"], extra["touch_teams"], extra["contested"]
            # touch tick colours for the timeline, one per frame (0 = none)
            self.touch_rgb = np.zeros((N_FRAMES, 3), np.uint8)
            for fr, tm, c in zip(self.tf, self.tt, self.tc):
                col = (170, 170, 170) if c else (C_HOME if tm == 0 else C_AWAY)
                self.touch_rgb[max(0, fr - 2):fr + 3] = col
        self.minimap_bg = self._minimap_background()

    # -- static pieces ---------------------------------------------------------------------
    def _mm(self, x, y):
        return self.MM_X0 + x * self.MM_W, self.MM_Y0 + y * self.MM_H

    def _minimap_background(self):
        img = Image.new("RGBA", (self.MM_W + 4, self.MM_H + 4), (20, 90, 40, 210))
        d = ImageDraw.Draw(img)
        line = (230, 230, 230, 255)
        sx, sy = self.MM_W / LENGTH, self.MM_H / WIDTH
        d.rectangle([2, 2, self.MM_W + 2, self.MM_H + 2], outline=line, width=2)
        d.line([2 + self.MM_W / 2, 2, 2 + self.MM_W / 2, self.MM_H + 2], fill=line, width=1)
        r = 9.15 * sx
        cx, cy = 2 + self.MM_W / 2, 2 + self.MM_H / 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=line, width=1)
        for x0, sign in ((2, 1), (2 + self.MM_W, -1)):
            bw, bh = 16.5 * sx, 40.32 * sy
            box = sorted([x0, x0 + sign * bw])
            d.rectangle([box[0], cy - bh / 2, box[1], cy + bh / 2], outline=line)
            gw, gh = 5.5 * sx, 18.32 * sy
            box = sorted([x0, x0 + sign * gw])
            d.rectangle([box[0], cy - gh / 2, box[1], cy + gh / 2], outline=line)
        return img

    # -- per frame ---------------------------------------------------------------------------
    def clock(self, f):
        for half, lo, hi in self.halves:
            if lo <= f <= hi:
                s = (f - lo) / FPS + (0 if half == 1 else 45 * 60)
                return "{} half  {:02d}:{:02d}".format("1st" if half == 1 else "2nd",
                                                       int(s // 60), int(s % 60))
        if f < self.halves[0][1]:
            return "before kickoff"
        if f > self.halves[-1][2]:
            return "after full time"
        return "halftime"

    def draw(self, frame_rgb, f):
        a = self.a
        arr = np.array(frame_rgb)
        if not self.clean:
            self._timeline(arr, f)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img, "RGBA")
        Hm = a["H"][f].astype(np.float64)
        ok_h = bool(a["valid"][f])

        players, team = a["players"][f], a["team"]
        ball = a["ball"][f]
        missing = np.isnan(ball[0])
        nearest = int(a["nearest"][f])

        # ---- on-video: players, ball, nearest line --------------------------------------
        if ok_h:
            vis = ~np.isnan(players[:, 0])
            px, py = load_homography.project(Hm, players[:, 0], players[:, 1])
            if not missing:
                bx, by = load_homography.project(Hm, ball[0], ball[1])
                if nearest >= 0 and vis[nearest]:
                    d.line([px[nearest], py[nearest], bx, by], fill=(255, 255, 255, 200), width=2)
                    d.text(((px[nearest] + bx) / 2 + 6, (py[nearest] + by) / 2 - 18),
                           "{:.1f} m".format(a["nearest_d"][f]), font=self.f_small,
                           fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            for s in np.flatnonzero(vis):
                col = C_HOME if team[s] == 0 else C_AWAY
                r = 9
                d.ellipse([px[s] - r, py[s] - r, px[s] + r, py[s] + r], fill=col + (230,),
                          outline=(255, 255, 255), width=2)
                d.text((px[s], py[s] - 22), str(s + 1), font=self.f_slot, anchor="mm",
                       fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            if not missing:
                trail = a["ball"][max(0, f - 25):f + 1]
                trail = trail[~np.isnan(trail[:, 0])]
                if len(trail) > 1:
                    tx, ty = load_homography.project(Hm, trail[:, 0], trail[:, 1])
                    d.line(list(zip(tx.tolist(), ty.tolist())), fill=C_BALL + (160,), width=3)
                d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=C_BALL, outline=(0, 0, 0),
                          width=2)
                d.ellipse([bx - 16, by - 16, bx + 16, by + 16], outline=C_BALL, width=2)
            if self.extra is not None:
                lo, hi = np.searchsorted(self.tf, [f - TOUCH_SHOW, f + TOUCH_SHOW + 1])
                for fr, tm, c in zip(self.tf[lo:hi], self.tt[lo:hi], self.tc[lo:hi]):
                    p = a["ball"][fr]
                    if np.isnan(p[0]):
                        continue
                    tx, ty = load_homography.project(Hm, p[0], p[1])
                    col = (190, 190, 190) if c else (C_HOME if tm == 0 else C_AWAY)
                    r = 30 - 1.5 * abs(fr - f)
                    d.ellipse([tx - r, ty - r, tx + r, ty + r], outline=col, width=5)
                    d.text((tx, ty - r - 14), "TOUCH" + (" (contested)" if c else ""),
                           font=self.f_small, anchor="mm", fill=col, stroke_width=2,
                           stroke_fill=(0, 0, 0))
            if missing and not np.isnan(a["last_ball"][f][0]):
                lx, ly = load_homography.project(Hm, *a["last_ball"][f])
                d.ellipse([lx - 14, ly - 14, lx + 14, ly + 14], outline=(200, 200, 200, 230),
                          width=3)
                d.line([lx - 10, ly - 10, lx + 10, ly + 10], fill=(200, 200, 200, 230), width=3)
                d.line([lx - 10, ly + 10, lx + 10, ly - 10], fill=(200, 200, 200, 230), width=3)
                d.text((lx, ly + 26), "last seen", font=self.f_small, anchor="mm",
                       fill=(220, 220, 220), stroke_width=2, stroke_fill=(0, 0, 0))

        # ---- missing-ball banner ----------------------------------------------------------
        if missing:
            el, tot = a["gap_elapsed"][f] / FPS, a["gap_total"][f] / FPS
            msg = "BALL NOT TRACKED   {:5.1f}s / {:.1f}s gap".format(el, tot)
            bw = d.textlength(msg, font=self.f_big) + 40
            d.rectangle([(W_IMG - bw) / 2, 14, (W_IMG + bw) / 2, 66], fill=(200, 20, 20, 215))
            d.text((W_IMG / 2, 40), msg, font=self.f_big, anchor="mm", fill=(255, 255, 255))

        # ---- HUD ---------------------------------------------------------------------------
        n_home = int(np.sum(~np.isnan(players[:, 0]) & (team == 0)))
        n_away = int(np.sum(~np.isnan(players[:, 0]) & (team == 1)))
        names = a["names"]
        lines = [
            ("frame {:6d}   {}".format(f, self.clock(f)), (255, 255, 255)),
            ("GRADED" if self.graded[f] else "not graded", C_OK if self.graded[f] else C_DEAD),
            ("tracked  {} {:2d}  {} {:2d}".format(names[0][:9], n_home, names[1][:9], n_away),
             (255, 255, 255)),
            ("ball     " + ("MISSING" if missing else "x {:5.1f} y {:4.1f} m".format(
                ball[0] * LENGTH, ball[1] * WIDTH)), C_MISS if missing else C_BALL),
            ("homog    " + ("reproj err {:.2f}".format(a["reproj"][f]) if ok_h else "INVALID"),
             (255, 255, 255) if ok_h else C_MISS),
        ]
        x0, y0, lh = 18, 14, 28
        n_label_rows = 0 if self.clean else 2 + (1 if self.extra is not None else 0)
        box_h = lh * (len(lines) + 1 + n_label_rows) + 16
        active = [e for e in self.events if e["start_frame"] <= f < e["end_frame"]]
        box_h += lh * min(len(active), 4)
        d.rectangle([x0 - 8, y0 - 6, x0 + 560, y0 + box_h], fill=(0, 0, 0, 215))
        y = y0
        for text, col in lines:
            d.text((x0, y), text, font=self.f_med, fill=col)
            y += lh
        rows = [("public GT", a["gt"][f]), ("baseline ", a["base50"][f])]
        if self.extra is not None:
            rows.append(("{:<9}".format(self.pred_name[:9]), self.pred[f]))
        if self.clean:
            rows = []
        for tag, lab in rows:
            d.text((x0, y), tag + "  ", font=self.f_med, fill=(255, 255, 255))
            d.rectangle([x0 + 165, y + 2, x0 + 195, y + 24], fill=LABEL_RGB[lab])
            d.text((x0 + 180, y + 13), lab, font=self.f_med, anchor="mm", fill=(255, 255, 255))
            y += lh
        d.text((x0, y), "smart tagging:" + ("" if active else "  -"), font=self.f_med,
               fill=(255, 255, 255))
        y += lh
        for e in active[:4]:
            t = e["tags"]
            extra = " ".join(v for k, v in t.items() if k in ("Team", "Type"))
            d.text((x0 + 16, y), "{} [{}]".format(e["code"].title(), extra)[:44],
                   font=self.f_small, fill=(255, 220, 120))
            y += lh

        # ---- minimap -----------------------------------------------------------------------
        img.paste(self.minimap_bg, (self.MM_X0 - 2, self.MM_Y0 - 2), self.minimap_bg)
        for s in np.flatnonzero(~np.isnan(players[:, 0])):
            mx, my = self._mm(*players[s])
            col = C_HOME if team[s] == 0 else C_AWAY
            d.ellipse([mx - 5, my - 5, mx + 5, my + 5], fill=col, outline=(255, 255, 255))
        if not missing:
            mx, my = self._mm(*ball)
            d.ellipse([mx - 4, my - 4, mx + 4, my + 4], fill=C_BALL, outline=(0, 0, 0))
        elif not np.isnan(a["last_ball"][f][0]):
            mx, my = self._mm(*a["last_ball"][f])
            d.ellipse([mx - 6, my - 6, mx + 6, my + 6], outline=(220, 220, 220), width=2)
        d.text((self.MM_X0, self.MM_Y0 - 26), "top-down (tracking coords)", font=self.f_small, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        return np.asarray(img)

    def _timeline(self, arr, f):
        """Three strips at the bottom, +/-30 s around now, drawn straight into the array."""
        lo, hi = f - self.TL_HALF, f + self.TL_HALF + 1
        idx = np.clip(np.arange(lo, hi), 0, N_FRAMES - 1)
        cols = np.linspace(0, len(idx) - 1, W_IMG - 240).astype(int)
        x0 = 220
        y = self.TL_Y0
        arr[y - 6:H_IMG, :] = (arr[y - 6:H_IMG, :] * 0.5).astype(np.uint8)
        strips = [self.ball_rgb, self.gt_rgb, self.base_rgb]
        if self.extra is not None:
            strips.append(self.pred_rgb)
        for rgb in strips:
            arr[y:y + self.TL_ROW - 4, x0:x0 + len(cols)] = rgb[idx[cols]][None, :, :]
            y += self.TL_ROW + 6
        if self.extra is not None:
            # touch ticks just above the prediction strip
            ty = y - (self.TL_ROW + 6) - 6
            ticks = self.touch_rgb[idx[cols]]
            on = ticks.any(axis=1)
            arr[ty:ty + 5, x0:x0 + len(cols)][:, on] = ticks[on][None, :, :]
        cx = x0 + len(cols) // 2
        arr[self.TL_Y0 - 6:H_IMG - 6, cx - 1:cx + 2] = 255

    def labels_layer(self, img_arr):
        if self.clean:
            return img_arr
        img = Image.fromarray(img_arr)
        d = ImageDraw.Draw(img)
        y = self.TL_Y0
        names = ["ball tracked", "public GT", "baseline ±2s"]
        if self.extra is not None:
            names.append(self.pred_name + " + touches")
        for name in names:
            d.text((18, y - 2), name, font=self.f_small, fill=(255, 255, 255))
            y += self.TL_ROW + 6
        d.text((220, H_IMG - 22), "-30 s", font=self.f_small, fill=(200, 200, 200))
        d.text((W_IMG - 80, H_IMG - 22), "+30 s", font=self.f_small, fill=(200, 200, 200))
        d.text((1080, H_IMG - 22),
               "ball: green=ok red=missing  label: blue=H red=A grey=D black=X",
               font=self.f_small, fill=(200, 200, 200))
        return np.asarray(img)


# ---------------------------------------------------------------------------------------------
# Video plumbing: ffmpeg decode -> numpy -> draw -> ffmpeg encode, one pipe pair per chunk.
# ---------------------------------------------------------------------------------------------

def load_extra():
    return dict(np.load(EXTRA)) if os.path.exists(EXTRA) else None


def render_chunk(job):
    start, count, out_path = job
    a = dict(np.load(CACHE))
    r = Renderer(a, load_smart_tagging.load(), load_extra())
    video = asset_paths()["video"]
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", "{:.6f}".format(start / FPS), "-i", video,
         "-frames:v", str(count), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "{}x{}".format(W_IMG, H_IMG), "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "3",
         "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)
    size = W_IMG * H_IMG * 3
    done = 0
    for k in range(count):
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H_IMG, W_IMG, 3)
        out = r.labels_layer(r.draw(frame, start + k))
        enc.stdin.write(out.tobytes())
        done += 1
    dec.stdout.close()
    dec.wait()
    enc.stdin.close()
    enc.wait()
    return start, done


def render(start, frames, name, workers):
    precompute()
    os.makedirs(OUT_DIR, exist_ok=True)
    parts_dir = os.path.join(OUT_DIR, "cache", name + "_parts")
    os.makedirs(parts_dir, exist_ok=True)
    chunk = int(np.ceil(frames / workers))
    jobs = [(s, min(chunk, start + frames - s), os.path.join(parts_dir, "{:06d}.mp4".format(s)))
            for s in range(start, start + frames, chunk)]
    t0 = time.time()
    total = 0
    with ProcessPoolExecutor(len(jobs)) as pool:
        for s, done in pool.map(render_chunk, jobs):
            total += done
            print("  chunk {:6d}: {} frames".format(s, done), flush=True)
    if total != frames:
        raise SystemExit("rendered {} frames, expected {}".format(total, frames))
    listing = os.path.join(parts_dir, "list.txt")
    with open(listing, "w") as handle:
        handle.writelines("file '{}'\n".format(j[2]) for j in jobs)
    out = os.path.join(OUT_DIR, name + ".mp4")
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                           "-i", listing, "-c", "copy", out])
    for j in jobs:
        os.remove(j[2])
    os.remove(listing)
    os.rmdir(parts_dir)
    print("{} frames in {:.0f}s -> {}".format(total, time.time() - t0, out))


def still(f):
    a = precompute()
    r = Renderer(a, load_smart_tagging.load(), load_extra())
    video = asset_paths()["video"]
    buf = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-ss", "{:.6f}".format(f / FPS), "-i", video, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    frame = np.frombuffer(buf, np.uint8).reshape(H_IMG, W_IMG, 3)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "still_{:06d}.png".format(f))
    Image.fromarray(r.labels_layer(r.draw(frame, f))).save(out)
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=None, help="default: to the end of the video")
    p.add_argument("--name", default="BAR-ATH_tactical_overlay")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--png", type=int, nargs="+", help="render single stills at these frames")
    p.add_argument("--labels", help="a submission CSV to show beside the baseline, with v3 touches"
                   " (e.g. evaluation/results/v3.csv)")
    args = p.parse_args()
    if args.labels:
        precompute_extra(args.labels)
    elif os.path.exists(EXTRA):
        os.remove(EXTRA)
    if args.png:
        for fr in args.png:
            still(fr)
    else:
        render(args.start, args.frames or N_FRAMES - args.start, args.name, args.workers)
