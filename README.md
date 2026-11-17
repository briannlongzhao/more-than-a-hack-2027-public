# Soma AI — More than a Hack 2027, Phase 1: Ball Possession

Our submission to [More than a Hack 2027](docs/CHALLENGE_README.md): for each of the 199,319
frames of FC Barcelona – Athletic Club, label who has the ball (`H` home, `A` away, `D` dead ball,
`X` no answer).

- **Submitted file:** [`submission/Soma_AI.csv`](submission/Soma_AI.csv) (version v11 of our pipeline)
- **Technical document:** [`submission/Soma_AI.md`](submission/Soma_AI.md)
- **Regenerate it:** `./reproduce.sh` (see below)
- **How it works, every version we tried, and results:** [`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md)

No human labels of any kind on this match were used. Every label comes from code, the public
5-minute answer key was used only to check the pipeline, and no thresholds were fitted to it.
The only learned component is [PathCRF](https://github.com/hyunsungkim-ds/pathcrf) (pinned as a
submodule), used with its published pretrained weights, which were trained on other matches.

## Method in brief

1. **Dead or live.** Players stop running during a stoppage, so low speed across the pitch and
   around the ball marks a stoppage, whether or not the ball is tracked. A tracked ball outside
   the pitch, or resting on a corner, centre or penalty spot, is dead. A set piece stays dead until
   the ball is kicked. Throw-ins restart when the ball is thrown back in. After a goal the ball is
   dead until the kick-off.
2. **Referee whistle.** A detector on the video's audio track finds the whistle tone near 3.8 kHz.
   A whistle starts a stoppage when play actually stops afterwards (no play-on).
3. **Which team.** PathCRF infers the team in possession from player movement alone, so it works
   on the ~41% of frames where the ball is not tracked. Possession only changes when the new team
   keeps the ball for ≥ 2 s or makes a clean, uncontested touch (detected from the ball's velocity
   change and player proximity). Each restart goes to the team that takes it.

Public 5-minute window: macroF1 0.907, spellF1 0.729. The full progression from the baseline is in
the guide.

## Reproducing the submission

Requirements: Linux, conda, ffmpeg, an NVIDIA GPU (CUDA 12.x driver), and the competition data
bundle, which is **not** in this repository because it is licensed to enrolled participants only.

```bash
git clone --recurse-submodules https://github.com/briannlongzhao/more-than-a-hack-2027-public.git
cd more-than-a-hack-2027-public

# data: unpack the bundle into ./data (or set MTAH_DATA_DIR)
#   data/tactical/BAR-ATH_tactical.mp4, BAR-ATH_atd.txt, BAR-ATH_atd.xml,
#   BAR-ATH_homography.csv, BAR-ATH_smart_tagging.xml

# environments
conda create -n soccer python=3.11 -y
conda run -n soccer pip install -r environments/soccer.txt
conda create -n pathcrf python=3.11 -y
conda run -n pathcrf pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n pathcrf pip install -r environments/pathcrf.txt

# everything: all versions, validation, public score, and submission/Soma_AI.csv
./reproduce.sh                       # PATHCRF_DEVICE=cuda:1 ./reproduce.sh to pick another GPU

sha256sum submission/Soma_AI.csv
# ac99c347f7012a53340e0170feee1e20985ccc966fd4a5e38e83c106021233e4
```

It takes about 1.5 minutes on one RTX 4090. The last step builds a local comparison website of
video clips; it contains match footage, so it stays on your machine and is never committed.

## Repository layout

| path | what |
|---|---|
| `solution/` | our pipeline: `features.py` (parsing), `v1.py` … `v11.py` (each version), `pathcrf_adapter.py`, `whistle.py`, `restart.py` |
| `submission/Soma_AI.csv` | the submitted file |
| `reproduce.sh` | one command that regenerates everything |
| `environments/` | pinned dependencies for the two conda environments |
| `tools/` | exploration plots, overlay renderer, comparison-website builder |
| `third_party/pathcrf` | PathCRF, pinned at commit `1b7c6f8` |
| `docs/` | our guide, and the organisers' original README |
| `starter/`, `evaluation/`, `games.py`, `*.md` at the top level | the organisers' starter kit, scorer and challenge documents, unchanged |

## Licence

Our code is MIT-licensed. The organisers' starter kit, scorer and documents keep their own MIT
licence (see [LICENSE](LICENSE)). PathCRF keeps its own licence. The match data is not covered
by any of these and is not included.
