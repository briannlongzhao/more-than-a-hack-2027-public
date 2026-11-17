"""Build a LOCAL web page comparing every version on sampled 20 s clips.

For each clip: the plain tactical video (no tracking overlay) and, underneath, one timeline of
output labels per version (public GT where it exists, baseline, v1..v5) with a cursor synced to
the video. Click a strip to seek; space = play/pause, left/right = one frame.

Clips are picked deterministically: four from the public 5 minutes (labels to compare against),
then the full-match windows where versions disagree most, plus two random ones as a control.

Output: data/artifact/compare/ (index.html + clips/). It contains match video, so it stays local
and under data/ (gitignored) -- the data terms forbid redistributing clips. Serve it with:

    python -m http.server 8765 --bind 127.0.0.1 --directory data/artifact/compare
"""

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
os.environ["OVERLAY_CLEAN"] = "1"
import render_overlay as R  # noqa: E402
import features as F  # noqa: E402
import v1 as V1  # noqa: E402

OUT = os.path.join(F.DATA_DIR, "artifact", "compare")
CLIP = 20 * F.FPS
RESULTS = os.path.join(ROOT, "evaluation", "results")
VERSIONS = [("baseline ±2s", "baseline_smooth50"), ("v1", "v1"), ("v2", "v2"), ("v3", "v3"),
            ("v4a", "v4a_pathcrf_all"), ("v4b", "v4b_pathcrf_gaps"), ("v5", "v5"), ("v6", "v6"), ("v7", "v7"), ("v8", "v8"), ("v9", "v9"), ("v10", "v10"), ("v11", "v11")]
NOTES = {
    "baseline ±2s": "nearest player, majority vote over ±2 s; ball missing = D",
    "v1": "speed-based dead ball + nearest player within 3 m for 0.8 s",
    "v2": "v1 dead ball + touches (velocity change or foot-dip); switch on any opposing touch",
    "v3": "v2 touches, but switch only on controlled, uncontested touches",
    "v4a": "v3 with PathCRF's team on every in-play frame",
    "v4b": "v3 with PathCRF's team only where the ball is missing",
    "v5": "PathCRF team + v3 switching discipline (change needs ≥2 s or a clean touch)",
    "v6": "v5, but set pieces stay D until the kick (ball rest→kick; ball resting ≥2 s on a spot)",
    "v7": "v6, plus a referee whistle followed by a stoppage starts the D at the whistle",
    "v8": "v6 + restart goes to the team of the first clean touch; live gaps <2 s between stoppages merged",
    "v9": "v8 + stopping whistle starts the D, unless play continues (a clean touch after it)",
    "v10": "v9 + throw-ins restart when the ball is thrown back into the pitch",
    "v11": "v10 + after a goal the ball is dead until the kick-off (Smart Tagging goal + kick-off)",
}


def load_labels(name):
    with open(os.path.join(RESULTS, name + ".csv")) as handle:
        next(handle)
        return np.array([line.strip().split(",")[1].upper() for line in handle], dtype="<U1")


def team_of(lab):
    return np.where(lab == "H", 0, np.where(lab == "A", 1, np.where(lab == "D", 2, -1)))


def pick_clips(labels, gt):
    graded = F.graded_mask()
    clips = [(s, "public 5 min (answer key available)") for s in (11400, 13600, 14700, 17300)]
    taken = [s for s, _ in clips]

    def free(s):
        return all(abs(s - t) >= 3 * CLIP for t in taken)

    def best(score, n, why):
        starts = np.arange(11185, 190197 - CLIP, CLIP // 2)
        starts = starts[[graded[s:s + CLIP].all() for s in starts]]
        vals = np.array([score(s) for s in starts])
        for s in starts[np.argsort(-vals, kind="stable")]:
            if len([c for c in clips if c[1] == why]) >= n:
                break
            if free(s) and score(s) > 0:
                clips.append((int(s), why))
                taken.append(int(s))

    v3, v5, v4a, base = (labels[k] for k in ("v3", "v5", "v4a_pathcrf_all", "baseline_smooth50"))
    best(lambda s: np.sum((v3[s:s + CLIP] != v5[s:s + CLIP]) & np.isin(v5[s:s + CLIP], ["H", "A"])),
         3, "v3 and v5 disagree on the team")
    best(lambda s: np.sum((base[s:s + CLIP] == "D") != (v5[s:s + CLIP] == "D")),
         2, "baseline and v1-v5 disagree on dead ball")
    best(lambda s: np.sum(v4a[s + 1:s + CLIP] != v4a[s:s + CLIP - 1]),
         2, "v4a switches most (fragmentation)")
    edges = os.path.join(F.DATA_DIR, "cache", "pathcrf_edges.npz")
    if os.path.exists(edges) and "v6" in labels:
        z = np.load(edges)
        out_fr = np.sort(z["frame"][np.char.startswith(z["dst"].astype(str), "out_")])
        v6 = labels["v6"]
        best(lambda s: np.sum(np.isin(out_fr, np.arange(s + 5 * F.FPS, s + CLIP - 5 * F.FPS)) &
                              np.isin(v6[np.clip(out_fr, 0, len(v6) - 1)], ["H", "A"])),
             2, "PathCRF says ball went out, v6 says live")
    if "v7" in labels and "v6" in labels:
        v6, v7 = labels["v6"], labels["v7"]
        best(lambda s: np.sum(v6[s:s + CLIP] != v7[s:s + CLIP]), 2, "whistle rule changes v6 → v7")
    if "v9" in labels and "v8" in labels:
        v8l, v9l = labels["v8"], labels["v9"]
        best(lambda s: np.sum(v8l[s:s + CLIP] != v9l[s:s + CLIP]), 2, "v9 whistle rule changes v8 → v9")
    if "v10" in labels and "v9" in labels:
        v9m, v10m = labels["v9"], labels["v10"]
        best(lambda s: np.sum(v9m[s:s + CLIP] != v10m[s:s + CLIP]), 3, "v10 throw-in rule changes v9 → v10")
    if "v11" in labels and "v10" in labels:
        a10, a11 = labels["v10"], labels["v11"]
        best(lambda s: np.sum(a10[s:s + CLIP] != a11[s:s + CLIP]), 3, "v11 goal rule changes v10 → v11")
    rng = np.random.default_rng(0)
    while len([c for c in clips if c[1] == "random"]) < 2:
        s = int(rng.integers(11185, 190197 - CLIP))
        if graded[s:s + CLIP].all() and free(s):
            clips.append((s, "random"))
            taken.append(s)
    return clips


def full_match_stats(lab):
    poss = sorted((s, e, t) for t in "HA" for s, e in V1.runs(lab == t))
    d = np.array([(e - s) / F.FPS for s, e, _ in poss])
    ff = sum(1 for i in range(1, len(poss) - 1) if d[i] < 2 and poss[i - 1][1] == poss[i][0]
             and poss[i][1] == poss[i + 1][0] and poss[i - 1][2] == poss[i + 1][2])
    return {"poss": len(d), "median": round(float(np.median(d)), 1), "flipflops": ff,
            "dead": round(float(np.mean(lab[F.graded_mask()] == "D")) * 100, 1)}


def public_scores():
    path = os.path.join(ROOT, "evaluation", "scoreboard.csv")
    out = {}
    if os.path.exists(path):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                name = row.get("submission") or row.get("name")
                out[name] = {k: row[k] for k in row if k in ("macro_f1", "spell_f1_mean")}
    return out


def main():
    os.makedirs(os.path.join(OUT, "clips"), exist_ok=True)
    labels = {key: load_labels(key) for _, key in VERSIONS}
    gt = load_labels_gt()
    clips = pick_clips(labels, gt)

    for old in os.listdir(os.path.join(OUT, "clips")):
        os.remove(os.path.join(OUT, "clips", old))
    jobs = [(s, os.path.join(OUT, "clips", "clip_{:06d}.mp4".format(s))) for s, _ in clips]
    with ProcessPoolExecutor(len(jobs)) as pool:
        for s in pool.map(cut_clip, jobs):
            print("clip {} cut".format(s), flush=True)

    data = {"fps": F.FPS, "versions": [n for n, _ in VERSIONS], "notes": NOTES, "clips": [],
            "stats": {n: full_match_stats(labels[k]) for n, k in VERSIONS},
            "public": {}}
    sb = public_scores()
    for n, k in VERSIONS:
        if k in sb:
            data["public"][n] = sb[k]
    for s, why in clips:
        e = s + CLIP
        data["clips"].append({
            "start": s, "n": CLIP, "why": why, "src": "clips/clip_{:06d}.mp4".format(s),
            "gt": "".join(gt[s:e]) if (gt[s:e] != "X").any() else None,
            "labels": {n: "".join(labels[k][s:e]) for n, k in VERSIONS},
        })
    with open(os.path.join(OUT, "index.html"), "w") as handle:
        handle.write(PAGE.replace("__DATA__", json.dumps(data)))
    print("->", os.path.join(OUT, "index.html"))


def cut_clip(job):
    """The plain tactical video for one clip -- no overlay, frame-accurate."""
    start, out_path = job
    subprocess.check_call(
        ["ffmpeg", "-v", "error", "-y", "-ss", "{:.6f}".format(start / F.FPS),
         "-i", R.asset_paths()["video"], "-frames:v", str(CLIP),
         # the stadium sound is recorded very quietly (~-59 dB): boost it so whistles are audible
         "-af", "volume=35dB,alimiter=limit=0.9", "-c:a", "aac", "-b:a", "96k",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", out_path])
    return start


def load_labels_gt():
    with open(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")) as handle:
        next(handle)
        return np.array([line.strip().split(",")[1].upper() for line in handle], dtype="<U1")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Possession Versions Compare</title>
<style>
:root{--bg:#111418;--panel:#1b2027;--text:#e8eaed;--muted:#9aa3ad;--line:#2c333c;
--H:#2b6eff;--A:#eb2832;--D:#8a8f96;--X:#2a2e33;--ok:#2fbf71;--miss:#e04848}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f5f6f8;--panel:#fff;
--text:#15181c;--muted:#5d6670;--line:#dde1e6;--X:#e3e6ea}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:20px 16px 8px;max-width:1280px;margin:auto}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted)}
main{max-width:1280px;margin:auto;padding:0 16px 40px}
table{border-collapse:collapse;width:100%;margin:12px 0 4px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.n{text-align:right}.note{color:var(--muted);font-size:12.5px}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:4px 8px}
nav{display:flex;flex-wrap:wrap;gap:6px;margin:16px 0}
nav button{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:999px;
padding:6px 12px;cursor:pointer;font:inherit;font-size:13px}
nav button.on{border-color:var(--H);box-shadow:inset 0 0 0 1px var(--H)}
.clip{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}
.cliphead{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;margin-bottom:8px}
.why{color:var(--muted)}video{width:100%;border-radius:8px;background:#000;display:block}
.stage{position:relative}.stage:fullscreen{background:#000;display:flex;align-items:center}
.stage:fullscreen video{border-radius:0;max-height:100vh;object-fit:contain}
.hud{position:absolute;top:2.5%;right:1.5%;background:rgba(0,0,0,.66);color:#fff;border-radius:8px;
padding:6px 8px;font:600 clamp(10px,1.35vw,18px)/1.25 ui-monospace,Menlo,Consolas,monospace;
display:grid;grid-template-columns:auto auto;gap:3px 10px;align-items:center;pointer-events:none}
.hud .lab{min-width:1.8em;text-align:center;border-radius:4px;padding:0 .3em;color:#fff}
.hud .gt{grid-column:1/3;height:1px;background:rgba(255,255,255,.35);margin:2px 0}
.hud .fr{grid-column:1/3;font-weight:400;opacity:.8;font-size:.85em}
.controls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0}
.controls button{background:transparent;color:var(--text);border:1px solid var(--line);border-radius:6px;
padding:4px 10px;cursor:pointer;font:inherit;font-size:13px}.controls button.on{border-color:var(--H)}
.time{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums}
.rows{display:grid;grid-template-columns:minmax(80px,120px) 1fr 34px;gap:4px 8px;align-items:center}
.rname{font-size:12.5px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
canvas{width:100%;height:18px;display:block;cursor:pointer;border-radius:3px}
.badge{font-weight:700;font-size:12px;text-align:center;border-radius:4px;padding:1px 0;color:#fff}
.legend{display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:12.5px;margin-top:10px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.loading{color:var(--muted);padding:40px;text-align:center}
</style></head><body>
<header><h1>Possession versions, side by side</h1>
<div class="sub">20-second clips from the tactical video. Timelines under the video show each version's label per frame; the white line is now. Click a strip to seek. Keys: space play/pause, ←/→ one frame.</div></header>
<main>
<div class="tablewrap"><table id="stats"></table></div>
<nav id="nav"></nav>
<section class="clip" id="clip"></section>
</main>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = {H:'--H',A:'--A',D:'--D',X:'--X'};
let cur = 0, video = null, blobUrls = {};

function statsTable(){
  const t = document.getElementById('stats');
  let h = '<tr><th>version</th><th>what it does</th><th class="n">public macroF1</th><th class="n">public spellF1</th><th class="n">full-match possessions</th><th class="n">median s</th><th class="n">flip-flops</th><th class="n">dead %</th></tr>';
  for (const v of DATA.versions){
    const s = DATA.stats[v], p = DATA.public[v] || {};
    const f = x => x === undefined ? '–' : (+x).toFixed(3);
    h += `<tr><td><b>${v}</b></td><td class="note">${DATA.notes[v]}</td><td class="n">${f(p.macro_f1)}</td><td class="n">${f(p.spell_f1_mean)}</td><td class="n">${s.poss}</td><td class="n">${s.median}</td><td class="n">${s.flipflops}</td><td class="n">${s.dead}</td></tr>`;
  }
  t.innerHTML = h;
}

function nav(){
  const n = document.getElementById('nav');
  n.innerHTML = '';
  DATA.clips.forEach((c, i) => {
    const b = document.createElement('button');
    b.textContent = `#${i+1} · ${c.why.split(' (')[0]}`;
    b.title = `frames ${c.start}–${c.start+500}`;
    b.className = i === cur ? 'on' : '';
    b.onclick = () => { cur = i; nav(); show(); };
    n.appendChild(b);
  });
}

function rowsFor(c){
  const r = [];
  if (c.gt) r.push({name:'public GT', kind:'lab', s:c.gt});
  for (const v of DATA.versions) r.push({name:v, kind:'lab', s:c.labels[v]});
  return r;
}

async function show(){
  const c = DATA.clips[cur], el = document.getElementById('clip');
  const n = c.n;
  el.innerHTML = `<div class="cliphead"><b>Clip #${cur+1} · frames ${c.start}–${c.start+n-1}</b><span class="why">${c.why}</span></div>
  <div class="stage" id="stage"><video id="v" playsinline preload="auto"></video><div class="hud" id="hud"></div></div>
  <div class="controls"><button id="pp">▶ play</button><button id="fs">⛶ fullscreen</button>
  <button data-r="0.25">¼×</button><button data-r="0.5">½×</button><button data-r="1" class="on">1×</button>
  <button id="back">◀ frame</button><button id="fwd">frame ▶</button><span class="time" id="tm"></span></div>
  <div class="rows" id="rows"></div>
  <div class="legend"><span><i class="sw" style="background:var(--H)"></i>H Barcelona</span><span><i class="sw" style="background:var(--A)"></i>A Athletic</span><span><i class="sw" style="background:var(--D)"></i>D dead ball</span><span><i class="sw" style="background:var(--X)"></i>X no label</span></div>`;
  video = document.getElementById('v');
  if (!blobUrls[c.src]){
    const r = await fetch(c.src); blobUrls[c.src] = URL.createObjectURL(await r.blob());
  }
  video.src = blobUrls[c.src];
  const rows = rowsFor(c), box = document.getElementById('rows'), canv = [];
  for (const r of rows){
    const name = document.createElement('div'); name.className = 'rname'; name.textContent = r.name;
    const cv = document.createElement('canvas'); const badge = document.createElement('div'); badge.className = 'badge';
    box.append(name, cv, badge); canv.push({r, cv, badge});
    cv.onclick = e => { const b = cv.getBoundingClientRect(); video.currentTime = (e.clientX - b.left) / b.width * n / DATA.fps; };
  }
  const draw = () => {
    const fr = Math.min(n - 1, Math.max(0, Math.round(video.currentTime * DATA.fps)));
    for (const {r, cv, badge} of canv){
      const w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
      if (cv.width !== w * dpr){ cv.width = w * dpr; cv.height = h * dpr; }
      const g = cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
      const px = w / n;
      if (r.kind === 'touch'){
        g.fillStyle = css('--X'); g.fillRect(0,0,w,h);
        for (const [f,t,con] of c.touches){ g.fillStyle = con ? '#aaa' : css(t===0?'--H':'--A'); g.fillRect(f*px-1,2,3,h-4); }
        badge.textContent = '';
      } else {
        let i = 0;
        while (i < n){ let j = i; while (j < n && r.s[j] === r.s[i]) j++;
          g.fillStyle = r.kind === 'ball' ? css(r.s[i]==='1'?'--ok':'--miss') : css(COL[r.s[i]] || '--X');
          g.fillRect(i*px,0,(j-i)*px+0.5,h); i = j; }
        if (r.kind === 'lab'){ const l = r.s[fr]; badge.textContent = l; badge.style.background = css(COL[l]); }
        else { badge.textContent = ''; }
      }
      g.fillStyle = '#fff'; g.fillRect(fr*px-1,0,2,h);
    }
    document.getElementById('tm').textContent = `frame ${c.start+fr} · ${(fr/DATA.fps).toFixed(2)} s`;
    let hud = `<span class="fr">frame ${c.start+fr}</span>`;
    for (const r of rows){
      const l = r.s[fr];
      hud += `<span>${r.name}</span><span class="lab" style="background:${css(COL[l])}">${l}</span>`;
      if (r.name === 'public GT') hud += '<span class="gt"></span>';
    }
    document.getElementById('hud').innerHTML = hud;
  };
  const loop = () => { if (!video) return; draw(); if (!video.paused) requestAnimationFrame(loop); };
  video.onplay = () => { document.getElementById('pp').textContent = '❚❚ pause'; loop(); };
  video.onpause = () => { document.getElementById('pp').textContent = '▶ play'; draw(); };
  video.onseeked = draw; video.onloadeddata = draw; window.onresize = draw;
  document.getElementById('pp').onclick = () => video.paused ? video.play() : video.pause();
  document.getElementById('fs').onclick = () => {
    const st = document.getElementById('stage');
    document.fullscreenElement ? document.exitFullscreen() : st.requestFullscreen();
  };
  document.addEventListener('fullscreenchange', draw);
  document.getElementById('back').onclick = () => { video.pause(); video.currentTime = Math.max(0, video.currentTime - 1/DATA.fps); };
  document.getElementById('fwd').onclick = () => { video.pause(); video.currentTime += 1/DATA.fps; };
  el.querySelectorAll('[data-r]').forEach(b => b.onclick = () => {
    video.playbackRate = +b.dataset.r; el.querySelectorAll('[data-r]').forEach(x => x.classList.toggle('on', x === b)); });
  draw();
}

document.addEventListener('keydown', e => {
  if (!video) return;
  if (e.code === 'Space'){ e.preventDefault(); video.paused ? video.play() : video.pause(); }
  if (e.code === 'ArrowRight'){ video.pause(); video.currentTime += 1/DATA.fps; }
  if (e.code === 'ArrowLeft'){ video.pause(); video.currentTime = Math.max(0, video.currentTime - 1/DATA.fps); }
});
statsTable(); nav(); show();
</script></body></html>"""


if __name__ == "__main__":
    main()
