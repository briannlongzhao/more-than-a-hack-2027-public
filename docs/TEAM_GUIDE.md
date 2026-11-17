# Team guide: what has been built, and how to reproduce it

This guide is for teammates joining the work. It covers what the pipeline does and why, where
things stand, and how to rebuild every result and the comparison website on your own machine.

For the challenge itself (task, labels, scoring, rules), read the repository's
[CHALLENGE_README.md](CHALLENGE_README.md), then [CHALLENGE.md](../CHALLENGE.md), [EVALUATION.md](../EVALUATION.md)
and [RULES.md](../RULES.md). In one sentence: for each of the 199,319 frames of one match
(Barcelona = **H**ome, Athletic Club = **A**way), output `H`, `A`, `D` (dead ball) or `X` (no
answer). The ranking metric is **spellF1**, which rewards recovering whole possessions, not just
getting individual frames right.

**Deadline: November 17, 2026.**

---

## 1. Where things stand

The current best version is **v11**. All numbers below are reproducible with one command (section 4).

| version | idea | public macroF1 | public spellF1 | full-match possessions | median length |
|---|---|---|---|---|---|
| baseline ±2 s | nearest player to the ball, majority vote; ball missing = D | 0.621 | 0.326 | 669 | – |
| v1 | dead ball from **player speed** + nearest player within 3 m for 0.8 s | 0.828 | 0.472 | 465 | 5.3 s |
| v2 | **touches** (ball velocity change or a foot next to the ball); switch on any opposing touch | 0.806 | 0.350 | 531 | 4.8 s |
| v3 | possession changes only on **controlled, uncontested** touches | 0.827 | 0.432 | 342 | 8.1 s |
| v4a / v4b | experiments: PathCRF's team everywhere / only where the ball is missing | 0.863 / 0.830 | 0.556 / 0.421 | 503 / 450 | 4.6 / 5.1 s |
| v5 | **PathCRF** decides *which* team, v3's discipline decides *when* it may change | 0.870 | 0.678 | 410 | 6.0 s |
| v6 | set pieces stay **D until the kick**; a ball resting ≥ 2 s on a corner/centre/penalty spot is D | 0.885 | 0.718 | 364 | 6.4 s |
| v7 | referee **whistle** starts the stoppage (dropped, see below) | 0.867 | 0.671 | 362 | – |
| v8 | the restart goes to the team of the **first clean touch**; live gaps < 2 s between stoppages merged | 0.885 | 0.718 | 346 | 6.6 s |
| v9 | stopping whistle starts the stoppage **unless play continues** | 0.907 | 0.729 | 345 | 6.6 s |
| v10 | throw-ins restart when the ball is **thrown back into the pitch** | 0.907 | 0.729 | 343 | 6.7 s |
| **v11** | after a **goal** the ball is dead until the kick-off | **0.907** | **0.729** | 326 | 6.7 s |

How to read this table:

- **"Public" = the 5-minute answer key in this repo** (frames 11,185–18,684, only 17 possessions).
  It is small and noisy: the scorer itself says gaps below ~0.3 spellF1 are within noise there.
  The real ranking uses a private full-match answer key we never see. Treat public scores as a
  sanity check, not a target.
- **v8–v11 change things that never happen in the public 5 minutes** (throw-ins, goals, injury
  whistles), so their public scores cannot move. They were checked by watching clips on the
  comparison website and with full-match statistics instead.
- **Full-match possessions / median length** need no answer key and are a useful plausibility
  check: real possessions in the public window average ~14 s, and flip-flopping output
  (H→A→H within a second or two) is almost always wrong.

### What was tried and dropped (worth a paragraph in the technical document)

- **v2's "switch on any opposing touch"**: an opponent's foot near the ball is not control. It
  fragmented possessions badly; v3 fixed it.
- **v4a, PathCRF alone**: best at naming the team, but it flips team too often. v5 combines it
  with v3's switching discipline.
- **v7, whistle without a check**: the whistle is real and detectable, but sometimes play goes on
  after a whistle (crowd whistling, advantage). v9 only uses a whistle when nobody plays the ball
  on afterwards.
- **PathCRF's "ball goes out" predictions**: only 17 of 93 matched our stoppages, so they're not used.
- **Corner-spot rule**: correct in principle, but the ball is basically never tracked while it
  sits at a corner, so it never fires. Corners are handled by the restart-kick rule instead.
- **Broadcast video**: not used. It isn't synchronised with the tactical video, which is the one
  the submission is indexed to.

### Known limitations (for the write-up)

- About 10% of public frames are still wrong. Almost all of it is **boundary timing of 1–3 s**:
  stoppages detected late, restarts detected late when the ball isn't tracked at the kick, and
  possession changes detected late.
- **Calm play near your own goal looks like a stoppage** (everyone stops running). Public clip #4
  shows the dead ball starting 2.9 s too early because of this.
- The ball is tracked on only **24% of dead-ball frames**, so many restarts fall back to "players
  speed up", which is less precise than seeing the kick.
- The Smart Tagging tagged one **disallowed goal** as a goal. v11 still handles it correctly
  (dead until the restart), because the score doesn't matter, only dead vs live.

---

## 2. How the pipeline works

Everything lives in `solution/`. Each version builds on the previous one and states its reasoning
in the docstring at the top of the file. Every threshold was set from football reasoning **before**
scoring, never fitted to the public 5 minutes (the rules forbid that).

**1. Inputs** (`features.py`): player and ball positions and speeds from the automatic tracking
(`*_atd.txt`), team of each tracking slot (`*_atd.xml`), and Smart Tagging set pieces and goals.
Positions are converted to metres on a 105.3 × 68 m pitch.

**2. Dead or live?** (`v1.py`, `restart.py`, `v8.py`–`v11.py`)
- **Stoppage start:** few players running anywhere **and** the 4 players nearest the ball walking,
  for ≥ 3 s, shifted 1.5 s earlier (measured: players go still ~1.3 s after a whistle). Also: the
  ball tracked outside the pitch, or resting ≥ 2 s on a restart spot.
- **Referee whistle** (`whistle.py`): a steady, narrow tone near 3.8 kHz in the video's audio. It
  starts the stoppage only if nobody plays the ball on afterwards (`v9.py`).
- **Stoppage end (restart):** the ball going from resting to fast with a player next to it (the
  kick), or a throw-in leaving the touchline fast (`v10.py`), or the kick-off after a goal
  (`v11.py`). If the ball isn't tracked, the moment the players speed up, with a lag measured from
  the restarts we can see.

**3. Which team?** (`v2.py`, `v3.py`, `pathcrf_adapter.py`, `v5.py`)
- **Touches** (`v2.py`): a sudden change in ball velocity with a player within 2 m, or a player's
  foot within 1 m of the ball.
- **PathCRF** (`pathcrf_adapter.py`): a published model (KDD 2026, trained on other matches) that
  infers who has the ball from **player movement only**, so it works when the ball isn't tracked
  (41% of frames). It's a git submodule in `third_party/pathcrf`. The adapter stitches our tracking
  fragments into ≤ 11 players per team and runs the pretrained model on the GPU.
- **Combination** (`v5.py`): PathCRF's team, but possession only changes when the new team has the
  ball for ≥ 2 s or makes a clean touch. Each restart goes to the team that takes it (`v8.py`).

---

## 3. Setting up (once)

### 3.1 Code

```bash
git clone --recurse-submodules https://github.com/briannlongzhao/more-than-a-hack-2027-public.git
cd more-than-a-hack-2027-public
# already cloned without submodules?  git submodule update --init
```

`third_party/pathcrf` must contain files, including `saved/140/`, the pretrained model.

### 3.2 Match data

The data is licensed to enrolled participants only. **Never commit it, upload it, or share
clips from it outside the team setup the terms allow.** Everything under `data/` is gitignored for
this reason.

Unpack the bundle so it looks like this (or point `MTAH_DATA_DIR` at it):

```
data/
├─ tactical/   BAR-ATH_tactical.mp4, _atd.txt, _atd.xml, _homography.csv, _smart_tagging.xml
└─ broadcast/  (optional, not used)
```

The download arrives as several zips plus separately downloaded videos, which may carry a `-001`
or `-002` suffix. Rename the video to exactly `BAR-ATH_tactical.mp4`. Check with:

```bash
python -c "from games import asset_paths; import os; print({k: os.path.exists(p) for k, p in asset_paths().items()})"
```

### 3.3 Environments

Two conda environments. They're kept separate because PathCRF needs PyTorch.

```bash
# main environment: everything except PathCRF
conda create -n soccer python=3.11 -y
conda run -n soccer pip install -r environments/soccer.txt

# PathCRF environment (needs an NVIDIA GPU + CUDA 12.x driver)
conda create -n pathcrf python=3.11 -y
conda run -n pathcrf pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n pathcrf pip install -r environments/pathcrf.txt
```

You also need **ffmpeg** on your PATH (used for the audio and the website clips). Tested with
ffmpeg 4.4.2.

> If `pip install` hangs, check your pip config (`pip config debug`) for an unreachable extra
> index. Adding `--isolated` makes pip ignore the config files.

---

## 4. Reproducing everything (one command)

```bash
./reproduce.sh
```

This rebuilds every version, then scores and validates the results, then builds the website:
baseline → v1–v3 → PathCRF → v4 → v5–v6 → whistle detection → v7–v11 → validate v11 →
score → website. It finished in about **1.5 minutes** on our machine (64 CPU cores, RTX 4090).
The first run is a little slower because it parses the tracking file (~200 MB) and decodes the
match audio once. Both are cached under `data/cache/`.

Options (environment variables):

| variable | default | when to change |
|---|---|---|
| `SOCCER_PY` | `<conda base>/envs/soccer/bin/python` | your envs live elsewhere |
| `PATHCRF_PY` | `<conda base>/envs/pathcrf/bin/python` | same |
| `PATHCRF_DEVICE` | `cuda:0` | pick another GPU, e.g. `cuda:1` |
| `MTAH_DATA_DIR` | `./data` | the data is elsewhere |

Outputs:

- `evaluation/results/<version>.csv`: one submission file per version (gitignored).
- `submission/Soma_AI.csv`: the file we submit, an exact copy of `v11.csv` (committed).
- `evaluation/scoreboard.csv` and the printed board: public 5-minute scores.
- `data/artifact/compare/`: the comparison website (gitignored, contains video).

**Reproducibility check:** running `./reproduce.sh` from scratch with all caches deleted produced
**byte-identical** files for every version (verified on Nov 14). The rules require the submitted
CSV to be regenerated *exactly*. The one sensitive step is PathCRF on the GPU: a different GPU
model or CPU inference could change the last digits and therefore a few labels. Run it on an
NVIDIA GPU and compare against a known-good file before submitting:

```bash
sha256sum evaluation/results/v11.csv
# reference (Nov 14, RTX 4090, torch 2.6.0+cu124):
# ac99c347f7012a53340e0170feee1e20985ccc966fd4a5e38e83c106021233e4
```

---

## 5. The comparison website

The website shows 20-second clips of the tactical video with each version's labels, so you can
see where a version is right or wrong. It's built by `tools/make_compare_site.py`, which runs as
the last step of `reproduce.sh`.

### Opening it

```bash
python -m http.server 8790 --bind 127.0.0.1 --directory data/artifact/compare
```

Then open http://localhost:8790. If it runs on a remote server, forward the port from your laptop
first:

```bash
ssh -L 8790:127.0.0.1:8790 <you>@<server>
```

It binds to `127.0.0.1` on purpose: the page contains match video, so don't expose it publicly
or upload it anywhere.

### What's on the page

- **Top table:** every version, a one-line description, public macroF1 and spellF1, and
  full-match possession count, median length, flip-flops and dead-ball share.
- **Clip buttons:** 25 clips, chosen automatically:
  - #1–4: the public 5 minutes, which have the answer key
  - then places where versions disagree most (team, dead ball, PathCRF vs v6)
  - clips where each later rule changed something (whistle v7/v9, throw-in v10, goal v11)
  - 2 random clips as a control
- **Under the video:** one timeline per version (blue = H, red = A, grey = D, dark = X), with a
  white cursor synced to the video. The answer key appears only on the public clips.
- **On the video (top-right):** each version's label for the current frame. The **⛶ fullscreen**
  button keeps this panel visible.
- **Controls:** click a timeline to seek, ¼× / ½× / 1× speed, **space** play/pause, **← →** one
  frame.
- **Sound** is on and boosted by +35 dB (the original track is very quiet), so you can hear
  referee whistles. Turn your volume down before the first click.

After changing a version, rebuild only the site with `python tools/make_compare_site.py`. Hard-refresh
the browser (Ctrl+Shift+R) so it doesn't show cached clips.

### Other visual tools (optional)

| command | output in `data/artifact/` |
|---|---|
| `python tools/render_overlay.py --start 17350 --frames 250 --name preview` | a clip with tracking drawn on the video (players, ball, nearest player, timeline) |
| `python tools/render_overlay.py --labels evaluation/results/v11.csv --png 15000` | one annotated still |
| `python tools/render_overlay.py --workers 16` | the full match with the overlay (~10 min, ~1.3 GB) |
| `python tools/plot_speed.py`, `plot_near_ball.py`, `plot_touch_detect.py`, `plot_audio.py` | the exploration plots behind each design decision |

---

## 6. Rules to keep in mind while working

- **No hand labels or hand corrections on this match**, ever, not even "fixing" one moment you
  watched. Watching clips to understand a *general* problem and then writing a *general* rule is
  fine. Writing down what happens at specific frames is not.
- **Don't tune thresholds to the public 5 minutes.** Use football reasoning, full-match statistics,
  and the website.
- **Never commit `data/`, result CSVs, clips or caches.** The `.gitignore` already covers them.
  Check `git status` before pushing anyway.
- Every rule needs a sentence of football reasoning in its docstring. That's what becomes the
  technical document.

---

## 7. What's left before November 17

| task | notes |
|---|---|
| ~~Pick the submission~~ | Done: v11, as `submission/Soma_AI.csv` (validated; `reproduce.sh` regenerates it). |
| **Technical document** | approach, assets used and dropped, training data (none of ours; PathCRF's pretrained weights come from public Bundesliga data), possession rules, limitations, reproduction. Sections 1–2 here are a starting point. |
| ~~Repo README for the submission~~ | Done: `README.md` in the public repo. |
| **Fresh-clone test** | one teammate follows this guide on a different machine and checks that `v11.csv` matches the `sha256sum`. |
| **Submit** | Google Form: team name, `<team_name>.csv`, repo URL, `<team_name>.pdf`. Needs a Google account. One response per team; the last one before the deadline counts. |
| **Team size** | 3–5 members are required for the later stages. |
