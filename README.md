# wallball-tracker

Computer-vision rep counter and cadence tracker for lacrosse wall-ball drills.
Runs locally on any laptop with a webcam; designed to deploy to a Raspberry Pi.

## What it does

- Tracks a colored ball with HSV blob detection + circularity filtering.
- Identifies real throws via parabolic-trajectory analysis with **four**
  confirmation paths (so reps register whether the ball leaves the frame,
  arcs in view, just keeps flying outbound, or only flashes briefly out of
  a lacrosse stick).
- Counts **outbound throws only** — the rebound flight back to the player
  is filtered out, so each wall-ball rep counts once.
- Reports rolling throws/min and detects drops two independent ways:
  - **Cadence-based** — current gap exceeds `2× median` of recent intervals.
  - **Position-based** — ball was last seen in the bottom of the frame
    (i.e. it fell to the floor) and tracking is lost.
- Logs every rep and drop to a per-session CSV.
- Has a clean hook for a pose estimator (MediaPipe / YOLO-pose) to slot in
  later without touching the rest of the pipeline.

## Files

| File | What it is |
|------|------------|
| `wallball.py` | Wall-ball rep counter — direction-aware, cadence, drops, CSV logging |
| `tracker.py`  | Simpler "is the ball being thrown" detector used during early R&D |

## Install

```bash
pip install opencv-python numpy
# Optional, on a Raspberry Pi with the CSI camera:
sudo apt install python3-picamera2
```

## Run

```bash
python wallball.py                       # webcam, wall on right
python wallball.py --wall-side left      # wall on left
python wallball.py --video clip.mp4      # offline replay
python wallball.py --no-display          # headless (Pi without monitor)
python wallball.py --log-dir my_sessions # custom log directory
```

Keys in the live window: `q` quit · `r` reset counts · `s` save screenshot.

## Rep confirmation paths

Every counted rep is tagged with which path confirmed it (visible in the
console and stored in the CSV via `pose_blob`/log lines):

| Tag    | Trigger |
|--------|---------|
| `OOF`  | Ball was tracked then disappeared (out of frame / motion blurred out / occluded). |
| `APEX` | Ball's visible trail crossed the fitted parabola's vertex with tight R²y. |
| `SUST` | Pending candidate stayed "outbound + fast" for `SUSTAINED_CONFIRM_S` (0.15s). Catches throws the camera tracks the whole way to the wall. |
| `BRST` | Short burst of fast outbound motion (≥ `MIN_BURST_POINTS` frames). For lacrosse-stick contexts where the ball is mostly hidden in the netting. |

## Drop detection

A drop fires when either signal triggers:

- **`CAD`**: gap since last rep > `max(1.5s, 2× median(last 4 intervals))`.
  Requires at least 3 reps to have established a rhythm.
- **`POS`**: ball was last seen below `GROUND_ZONE_Y_FRAC` of the frame
  (default 78%, i.e. bottom 22%) and tracking has been lost for ≥ 0.4s.

When both fire on the same event it's tagged `BOTH`.

## Pose-estimation hook

`PoseEstimator` in `wallball.py` is a no-op base class. To enable:

```python
class MediaPipePose(PoseEstimator):
    enabled = True
    def __init__(self):
        import mediapipe as mp
        self.solver = mp.solutions.pose.Pose(...)
    def __call__(self, frame):
        # return dict, e.g.:
        # {"shoulder_angle_deg": 95.2, "elbow_angle_deg": 110.3,
        #  "release_height_px": 180.0}
        ...
    def annotate(self, frame, info):
        ...
```

Then change one line in `main()`:

```python
pose = MediaPipePose()
```

The main loop calls it every frame; `pose.serialize(info)` gets stamped
into the `pose_blob` column of every rep so you can correlate form metrics
with cadence later.

## Tuning

Top-of-file constants in `wallball.py`. The dials you'll touch most:

- `HSV_LOW_A/B`, `HSV_HIGH_A/B` — ball color range. Two passes OR'd:
  a saturated body range, plus a tight bright-specular-highlight range.
- `MIN_PEAK_SPEED_PX`, `MIN_OUTBOUND_VX_PX` — speed gates. Auto-scaled by
  detected ball radius so distance changes don't break detection.
- `REP_COOLDOWN_S` — suppress double-counting of one physical throw.
- `DROP_CADENCE_RATIO`, `GROUND_ZONE_Y_FRAC` — drop sensitivity.

## Hardware notes

The current pipeline works on a 30 fps webcam, but the speed and motion-blur
limits show. A **global-shutter camera at 100+ fps** removes motion blur,
gives many more trajectory points per rep, and lets every gate be tightened.
