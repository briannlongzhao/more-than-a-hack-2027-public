# Soma AI — Technical document

More than a Hack 2027, Phase 1: Ball Possession · FC Barcelona – Athletic Club

- **Submission:** `submission/Soma_AI.csv` (pipeline version v11)
- **Code:** https://github.com/briannlongzhao/more-than-a-hack-2027-public
- **Regenerate:** `./reproduce.sh` (sha256 `ac99c347f7012a53340e0170feee1e20985ccc966fd4a5e38e83c106021233e4`)

## 1. Approach

For each frame we answer two questions in order: **is the ball in play?** and, if it is,
**which team has it?** Both answers come from rules over the provided automatic data, plus one
pretrained public model (PathCRF). No model was trained on this match, and no label on this match
came from a person.

Why rules instead of a trained model: we have no labelled possession data for this match, and
the rules forbid creating any. The other option, training on public data from other games, would
have needed a whole tracking pipeline to bridge the domain gap. The hard parts here are specific
events, such as a ball that isn't tracked, a whistle, a throw-in or a celebration. Each has a
clear physical signature that we could write a rule for and check by watching the video.

**The main obstacle:** the ball is untracked on about 41% of graded frames, and on 76% of
dead-ball frames. So the pipeline never relies on the ball alone. It uses the 22 players'
movement, the audio and the Smart Tagging.

### 1.1 Dead or live

- **Players stop.** A frame is "still" when fewer than 15% of tracked players run faster than
  2 m/s and the 4 players nearest the (last known) ball move slower than 1.5 m/s, each measured
  as a 1 s mean. A still stretch of at least 3 s is a stoppage. Its start moves 1.5 s earlier,
  because players slow down only after the ball goes dead.
- **Ball out.** A tracked ball more than 0.5 m outside the pitch is dead. A ball resting for at
  least 2 s on a corner, centre or penalty spot is dead (a set piece being prepared).
- **Referee whistle.** The tactical video's audio is quiet but contains the whistle: a steady,
  narrow tone at 3.55–4.05 kHz. We take an STFT with one hop per video frame and look for the
  peak that stands out from the 3.0–4.6 kHz background. A frame counts if the peak is at least
  15 dB above that background and its frequency is steady. An event is a whistle if its peak is
  at least 28 dB and the tone is sharp: at least 10 dB above its neighbouring frequencies. A
  whistle in live play starts the stoppage only if play actually stops within 8 s and nobody
  plays the ball on in between. This filters out crowd whistling and played advantages.
- **Restart.**
  - *Kick:* the tracked ball goes from resting (<2.5 m/s) to struck (>6 m/s) with a player
    within 2 m.
  - *Throw-in:* the ball comes back from beyond the touchline and then leaves the 3 m touchline
    zone fast. The ball being held over the line is not the restart.
  - *Goal:* from the last ball sighting in the penalty box to the kick-off. We find the goal and
    the kick-off from the Smart Tagging "Goal" and "Kick Off (After Goal)" tags, and refine both
    on the tracked ball.
  - *No tracked ball:* when none of these is visible, the stoppage ends when the players speed
    up again, minus a lag. The code measures that lag automatically at run time, on the restarts
    where the kick is visible.
- **Clean-up.** Live gaps shorter than 2 s between two stoppages are merged into one stoppage.

### 1.2 Which team

- **Touches** (`v2.py`). A touch is a change in ball velocity of at least 3 m/s (velocities
  compared over 0.32 s on each side) with a player within 2 m, or the ball within 1 m of a
  player.
- **Contested touches don't count** (`v3.py`). If players of both teams are within 1.5 m of the
  ball, the tracking cannot say who touched it.
- **PathCRF** (`pathcrf_adapter.py`). This published model infers the possession state from
  player trajectories only (Kim et al., *PathCRF: Ball-Free Soccer Event Detection via
  Possession Path Inference from Player Trajectories*, KDD 2026). It therefore works when the
  ball is untracked.
  - Our adapter stitches the 46 tracking slots into at most 11 continuous players per team and
    interpolates short gaps. It then rescales the coordinates to PathCRF's pitch and runs the
    published *Dynamic Masked CRF* checkpoint (trial 140) at 5 Hz.
  - We read the team of the player in possession from each predicted edge.
- **Combining them** (`v5.py`). PathCRF decides *which* team; where it gives no answer, the
  touch-based team from `v3.py` fills in. Possession switches only when the
  other team's run lasts at least 2 s or contains a clean, uncontested touch by that team.
  Shorter flickers are ignored.
- **Restarts** (`v8.py`). A live stretch starts with the team of the first clean touch within
  2 s of the restart. Without one, it starts with the team evidence on its first frame (PathCRF,
  else touches, else the Smart Tagging set-piece team).

## 2. Assets: used and dropped

| asset | use |
|---|---|
| Automatic tracking (ATD) | everything: player speeds, ball velocity, touches, input to PathCRF |
| Smart Tagging | set-piece team as a fallback; goal and kick-off tags (v11). The FAQ confirms these are automatic, not human |
| Tactical video audio | referee whistle detection |
| Tactical video frames | only for watching our own output while debugging (allowed by the rules); not an input |
| Homography | not used. ATD is already in pitch coordinates |
| Broadcast video | not used. It is not synchronised with the tactical video the submission is indexed to |

**Tried and dropped:**
- **Switching on any opposing touch (v2).** A foot near the ball is not control, and this
  fragmented possessions.
- **PathCRF alone (v4a).** It names the team well but flips too often.
- **An unconditional whistle rule (v7).** It fired on whistles where play continued.
- **PathCRF's "ball out" predictions.** Only 17 of 93 matched a stoppage.
- **A pose-based idea on the broadcast video.** Too costly for the gain, and the broadcast
  video is unsynchronised.

## 3. Training data

**We labelled nothing and trained nothing.** The only learned component is PathCRF's published
pretrained checkpoint, used unchanged (`third_party/pathcrf`, commit `1b7c6f8`, MPL-2.0).

The PathCRF authors trained it on the **Sportec Open DFL Dataset** (Bassek et al., 2025, *Scientific
Data*): event and tracking data from **7 German Bundesliga (1st and 2nd division) matches**, none
of them the provided game.
- **Who labelled it:** the event data was annotated manually by the league's official data
  provider (DFL).
- **How the labels were made:** the PathCRF authors synchronised the events with the tracking
  using ELASTIC (Kim et al., 2025), then derived per-frame possession states from them.

Nothing from the provided game, and no human-produced data about it, was used at any point.
The public 5-minute answer key was used only to check that the pipeline works, never to choose
thresholds. Every threshold above was set from football reasoning, or measured by code at run
time.

## 4. Possession rules we implemented

- **Loose balls and passes in flight** keep the last team in control. A pass to a teammate is
  never a change.
- **A change needs control.** The new team must keep the ball (≥ 2 s of PathCRF evidence) or
  make a clean, uncontested touch. Deflections, blocks and lost 50/50 duels are not changes. A
  confirmed change is dated back to where the new team's evidence began.
- **Contested balls**, with both teams within 1.5 m, don't change possession.
- **Restarts** are dead (`D`) until the ball is played: free kicks, corners and goal kicks
  until the kick, throw-ins until the throw, and goals until the kick-off. The restart then
  belongs to the team that takes it.
- **Injuries and other stoppages** are dead from the referee's whistle if play stops.
- **We never output X on graded frames.** The scorer would fill X with our last label anyway.

## 5. Results and limitations

Public 5-minute window: **macroF1 0.907, spellF1 0.729**. For comparison, the provided baseline
scores 0.621 and 0.326. Over the full match the output has 326 possessions, with a median length
of 6.7 s.

Where it fails:
- **Boundary timing of 1–3 s** causes most of the remaining error: late stoppages, late
  restarts when the ball isn't tracked at the kick, and late possession changes.
- **Calm play near a team's own goal** looks like a stoppage, because everyone stops running.
- **Restarts with an untracked ball** fall back to "players speed up". That is less precise
  than seeing the kick.
- **Whistles** are missed when crowd noise masks them, because the audio is very quiet.

With more time we would:
- learn the dead-ball boundaries from the Sportec data (other matches) rather than setting them
  by hand;
- fine-tune PathCRF on tracking with the same noise as this provider's;
- add a ball detector on the tactical video to fill the gaps in ball tracking.

## 6. Reproduction

```bash
git clone --recurse-submodules https://github.com/briannlongzhao/more-than-a-hack-2027-public.git
cd more-than-a-hack-2027-public
# put the data bundle in ./data (or set MTAH_DATA_DIR); create the two conda envs:
conda create -n soccer python=3.11 -y && conda run -n soccer pip install -r environments/soccer.txt
conda create -n pathcrf python=3.11 -y
conda run -n pathcrf pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n pathcrf pip install -r environments/pathcrf.txt
./reproduce.sh                  # about 1.5 min on one NVIDIA RTX 4090; needs ffmpeg
sha256sum submission/Soma_AI.csv
```

We checked this from a fresh clone: the output was byte-identical. PathCRF inference runs on the
GPU, so a different GPU model could change some floating-point results. The reference run used
an RTX 4090 with torch 2.6.0+cu124. Full details are in `docs/TEAM_GUIDE.md`.
