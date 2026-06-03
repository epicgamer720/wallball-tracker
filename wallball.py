"""
Wall-ball rep counter.

Counts only OUTBOUND throws (ball traveling toward the wall), tracks rolling
cadence, detects drops, and logs every event to a CSV.  Architecture leaves
a clean seam for a pose estimator to slot in later without rewrites.

Pipeline:
  frame -> HSV mask (two-range, OR'd) -> morph -> best ball contour
        -> (t, x, y, r) trail -> longest suffix that fits a parabola AND
           moves toward the wall -> release confirmation -> rep counted
        -> cadence + drop tracking -> CSV log -> HUD overlay

Wall-ball specifics vs. the generic tracker:
  - Direction gate: ball x-velocity must be in the wall direction.
    A real rep is throw + rebound; we only want to count one of them.
  - Cadence: rolling throws/min over CADENCE_WINDOW_S.
  - Drop detection: gap > DROP_THRESHOLD_S since last rep = a drop.
  - SessionLogger: timestamped CSV, one row per rep + drop, summary footer.
  - PoseEstimator: stub. Replace with MediaPipe / YOLO-pose (see class doc).

Run:
    python wallball.py
    python wallball.py --wall-side left
    python wallball.py --video clip.mp4
    python wallball.py --no-display          # headless

Keys (when displaying):
    q  quit       r  reset counts       s  save annotated screenshot
"""
import argparse
import collections
import csv
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime

# Tell numpy/BLAS to use all cores BEFORE numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS",        "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "8")
os.environ.setdefault("MKL_NUM_THREADS",        "8")

import numpy as np
import cv2


# --- Tunables ------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480    # sweet spot: ~4x lighter than 1280, smoother tracking
TARGET_FPS       = 120         # OV9782 does 640x480@120; more frames = denser trail
DISPLAY_EVERY_N  = 1           # render every frame (cheap at 30fps capture).
                               # Bump up if pushing 100+ fps capture so the
                               # display thread stops being the bottleneck.

# Anti-jitter: One-Euro filter (Casiez et al, 2012).  Adaptive smoothing
# that's *heavy* on slow / stationary input (kills pixel-edge noise) and
# *light* on fast input (no lag on real throws).  fc_min is the minimum
# cutoff in Hz; beta scales cutoff with motion speed.  Lower fc_min =
# smoother when still; higher beta = less lag when moving fast.
ONEEURO_FC_MIN          = 1.0     # Hz (position channels cx, cy)
ONEEURO_BETA            = 0.7     # Hz / (px/s)
ONEEURO_R_FC_MIN        = 0.4     # radius changes slowly; smooth more
ONEEURO_R_BETA          = 0.1
FILTER_RESET_GAP_S      = 0.20    # only reset filters after this much absence

# Core/thread layout. The Pi has 8 cores. With the mask/contour stage now
# running on its own worker thread, cv2 ops happen from two threads (the
# main thread's drawing + the worker's HSV/morph/findContours). Let cv2
# parallelize aggressively across all cores.
CV_INTERNAL_THREADS = 8

# Yellow ball — two-range HSV (saturated body OR bright specular highlight).
HSV_LOW_A  = np.array([15, 100,  60], dtype=np.uint8)
HSV_HIGH_A = np.array([42, 255, 255], dtype=np.uint8)
HSV_LOW_B  = np.array([15,  20, 235], dtype=np.uint8)
HSV_HIGH_B = np.array([42,  95, 255], dtype=np.uint8)

MIN_RADIUS_PX     = 5          # LOOSE: only used in tracking mode (we have an anchor)
MAX_RADIUS_PX     = 100        # human torsos in frame are >>100px — reject by size
MIN_CIRCULARITY   = 0.40       # LOOSE-mode floor.  A finger across the ball
                               # can drop circularity to ~0.5, so we need to
                               # let those through.  Roundness still dominates
                               # the score via the circ^2 term below.

# STRICT gates for re-acquire mode (no anchor — we just lost the ball or
# haven't found it yet).  At this point we don't want every yellow speck
# competing for "biggest score" — we want only an obvious ball-shaped
# blob to qualify.  Once acquired, the loose gates above kick in so
# partial occlusion doesn't drop us back into re-acquire.
#
# 22px radius -> ~1500 px^2 minimum area; tiny background specks are
# rejected outright, even if they're perfectly round.  Combined with the
# `circ * area` score below, a real ball crushes any plausible noise blob.
MIN_RADIUS_PX_STRICT    = 22
MIN_CIRCULARITY_STRICT  = 0.60

# Absolute minimum ball radius (full-res px), enforced in BOTH tracking and
# re-acquire modes.  Stops the tracker latching onto tiny specks and keeps the
# anchor from ratcheting down to nothing when it loses the ball.  Set per
# resolution preset (and tunable live from the Setup tab).
MIN_BALL_RADIUS_PX      = 12
TRAIL_MAX_AGE_S   = 1.0
TRAIL_LEN         = 240        # generous for high-fps cameras

# Position-aware selection.  When we have a recent detection:
#   - HARD REJECT contours whose center is farther than TRACK_HARD_PX
#     from the last ball position.  Kills "yellow speck across the frame
#     becomes the ball" entirely.
#   - HARD REJECT contours that GREW by more than TRACK_GROW_TOL — a
#     real ball doesn't suddenly get bigger, but partial occlusion makes
#     it look smaller, so shrinking is allowed (down to TRACK_SHRINK_TOL).
#   - Soft proximity bonus for continuity (keeps the closest blob winning).
# Strict-tracking anchor is dropped after TRACK_LOST_GAP_S.  After that,
# a softer "ghost anchor" (last known position only, no size lock) keeps
# providing a spatial prior for up to GHOST_ANCHOR_GAP_S so re-acquire
# prefers the ball where we last saw it instead of glueing onto a
# distant spec.
TRACK_HARD_PX           = 120.0  # 640-wide frame; rejects far-off blobs while tracking
TRACK_GROW_TOL          = 0.35   # reject contours that grew >35% vs anchor
TRACK_SHRINK_TOL        = 0.60   # allow shrinking down to 40% of anchor (partial occlusion)
                                 # — tighter than 0.85 so the tracker can't
                                 # latch onto a small chunk of body/clothing
                                 # near the last position.
TRACK_SEARCH_PX         = 110.0  # scale of soft proximity bonus
TRACK_BONUS             = 3.0
TRACK_LOST_GAP_S        = 0.30

# Anchor coasting + speed-aware gate (throw survival).  A fast throw motion-
# blurs the ball, so single-frame detection drops out for a few frames.  Rather
# than freezing the anchor where the ball vanished (then falling to strict
# re-acquire), keep the loose-gated anchor alive a little longer and slide it
# along the ball's last measured velocity, so the search region follows the
# ball through the blur and the loose gates can re-grab it on the far side.
COAST_EXTRA_S       = 0.12   # keep the loose-gated anchor this long past TRACK_LOST_GAP_S
COAST_MAX_S         = 0.12   # cap how far ahead to extrapolate (constant-v is only good briefly)
COAST_VEL_SPAN_S    = 0.04   # estimate velocity over >= this much trail history (de-noise)
TRACK_JUMP_K        = 1.5    # allow per-frame jumps up to K x (speed x time-since-last-fix)
TRACK_JUMP_MAX_PX   = 300.0  # absolute cap on the widened hard gate (px)

# Ghost-anchor (position-only memory after the strict anchor expires).
# Survives brief occlusion — when the ball emerges, the visible blob near
# the last-known location wins via proximity bonus.  Tuned softer than
# the strict tracking anchor: if the bonus is too strong, walking in
# front of the camera makes the tracker glue onto body/clothing right
# where the ball was.
GHOST_ANCHOR_GAP_S      = 2.0
GHOST_ANCHOR_RANGE_PX   = 130.0
GHOST_ANCHOR_BONUS      = 2.0    # mild bias, not a takeover
GHOST_REACQUIRE_CIRC    = 0.50   # circularity gate while a ghost anchor is live
                                 # — between strict (0.60) and loose (0.40).
                                 # Occlusion/blur knocks a reappearing ball's
                                 # circularity down; the proximity prior already
                                 # guards against distant noise, so relax here so
                                 # the ball can re-lock near where it vanished.
                                 # Radius stays strict (a half-occluded ball
                                 # keeps its enclosing-circle radius, so radius
                                 # is the right speck filter to retain).  Push
                                 # toward 0.40 if recovery is still poor.

# Fragment-fit fallback: when the ball is heavily occluded by a lacrosse
# stick mesh or a hand, no single contour looks like a ball.  Instead the
# ball shows up as one C-shape (finger across ball) or N small fragments
# (mesh).  Whatever the geometry, the visible ball EDGE points all lie on
# the ball's circle — we just have to ignore the non-edge points (finger
# chord, interior speckle, etc.) when fitting.
#
# Strategy: from every contour, keep only points whose distance from the
# anchor center lies in the "expected edge band" (FRAG_EDGE_INNER_R to
# FRAG_EDGE_OUTER_R times anchor.r).  Those are real ball-arc points.
# Points on a finger chord lie INSIDE the ball circle (closer than the
# inner-band cutoff) and get dropped — they no longer bias the fit.
#
# Random yellow specks don't lie on a common circle, so even if a few of
# their points fall in the band, the residual check rejects them.
#
# Only runs when (anchor is set) AND the normal best-contour pick was
# missing or smaller than the expected ball.
FRAGMENT_FIT_TRIGGER_R  = 0.85   # trigger fallback if best.r < anchor.r * this
FRAGMENT_MIN_AREA       = 6      # ignore tiny noise specks
FRAGMENT_MIN_POINTS     = 6      # need >= this many edge-band points to fit
FRAG_EDGE_INNER_R       = 0.85   # keep points with dist >= this * anchor.r
                                 # (tight enough to drop most finger-chord
                                 # points, loose enough for ~10% anchor.r
                                 # uncertainty + pixel discretization)
FRAG_EDGE_OUTER_R       = 1.15   # keep points with dist <= this * anchor.r
FRAGMENT_CENTER_TOL     = 0.45   # fitted center within anchor.r * this of anchor
FRAGMENT_RADIUS_TOL     = 0.30   # fitted radius within 30% of anchor radius
FRAGMENT_RESIDUAL_PX    = 4.0    # median |dist-to-circle| must be <= this

# Loose-lock mode (opt-in via the Debug toggle).  Trades a little false-positive
# risk for not losing the ball: drops the circularity gate (so blurry/partial
# shapes still qualify) and, when no clean contour passes at all, falls back to
# the CENTROID of the right-coloured mass within a window around the anchor.
# That leverages "there's a smear of the right colour here, just not round
# enough for a normal lock" to keep tracking through fast motion.  The window +
# fill threshold stop it latching onto stray colour far from the last position.
LOOSE_LOCK_CIRC      = 0.20      # circularity gate while loose-lock is on (anchor/ghost only)
BLOB_WIN_R           = 2.6       # colour-blob search half-window = this * anchor.r
BLOB_WIN_MIN_PX      = 26.0      # ...but at least this half-width (detection-res px)
BLOB_MIN_FILL        = 0.18      # need >= this fraction of the anchor disc filled with colour

# Parabolic-fit thresholds — loosened so fast, blurry, near-straight wall
# ball throws still produce a candidate.  The speed gates below do the real
# filtering of fake/idle motion.
MIN_POINTS_FIT    = 7
MAX_FIT_PTS       = 36       # cap the suffix-window scan — bounds per-frame cost
                            # at high fps; a wall-ball throw fits well within this
MIN_DURATION_S    = 0.10
MIN_TOTAL_DISP_PX = 120
MIN_A_PX_S2       = 250.0
MIN_R2_Y          = 0.70       # was 0.85
MIN_R2_X          = 0.55       # was 0.80
REF_RADIUS_PX     = 30.0       # tuned for 640x480 frame size above
MIN_SCALE         = 0.3

# Speed gates — real throws blow past these; carrying/idle gestures don't.
# Field data (2026-05-31): real throws peak >1500 px/s (typically 2500-5500);
# a ball *carried* toward the wall peaked <700.  The gate scales with ball
# size, but MIN_SCALE=0.3 dropped it to ~150 for small/distant balls and let
# slow carries through (then SUST/burst confirmed them).  So peak speed now
# also has an absolute floor (MIN_PEAK_SPEED_FLOOR_PX) that scaling can't drop
# below.  Lower the floor to count soft tosses; raise it if carries still log.
MIN_PEAK_SPEED_PX       = 900.0     # was 500; scaled by ball size
MIN_PEAK_SPEED_FLOOR_PX = 800.0     # absolute floor on peak speed, any ball size
MIN_OUTBOUND_VX_PX      = 250.0     # was 100

# Burst detector — for lacrosse-stick throws, where the ball is only briefly
# visible mid-flight (it lives in the stick's netting otherwise). Fires
# without a parabolic fit when there's a short burst of fast outbound motion.
MIN_BURST_POINTS        = 3      # was 4 — catch very brief lacrosse releases
MIN_BURST_DURATION_S    = 0.03
MIN_BURST_DX_PX         = 80     # at REF_RADIUS_PX; scaled by ball size
MIN_BURST_SPEED_PX      = 1200.0 # was 700; real bursts peak >3000. Also floored
                                 # by MIN_PEAK_SPEED_FLOOR_PX (see speed gates).
MIN_OUTBOUND_FRAC       = 0.7    # frame-to-frame x deltas must mostly point at wall

# After a rep is counted, suppress new confirmations for this long. Prevents
# one physical throw from firing twice (e.g. outbound -> OOF, then rebound +
# next-throw mix -> APEX a half-second later). Real back-to-back lacrosse
# throws are typically >= 0.6s apart.
REP_COOLDOWN_S          = 0.50

# Release confirmation — three paths:
#   OOF  ball disappears
#   APEX visible arc with apex inside the trail and tight R^2
#   SUST ball keeps moving fast outbound for SUSTAINED_CONFIRM_S after
#        the candidate opens.  Catches real throws that the camera tracks
#        all the way to the wall (no OOF, no clean apex).
PENDING_WINDOW_S    = 0.8
RELEASE_GAP_S       = 0.08
MIN_R2_Y_INFRAME    = 0.88
APEX_EDGE_MARGIN_S  = 0.05
LABEL_HOLD_S        = 0.6
SUSTAINED_CONFIRM_S = 0.15      # how long pending stays "thrown+outbound" to confirm

# Cadence + drops.
CADENCE_WINDOW_S    = 30.0
# Cadence-relative drop detection: a drop fires when the current gap since
# the last rep exceeds DROP_CADENCE_RATIO * (median of recent intervals),
# with a hard floor of DROP_MIN_GAP_S to avoid noise at very fast tempos.
# Needs DROP_MIN_REPS reps to have established a rhythm first.
DROP_CADENCE_RATIO  = 2.0
DROP_MIN_GAP_S      = 1.5
DROP_MIN_REPS       = 3
DROP_LOOKBACK       = 4         # use the last N intervals for the median

# Position-based drop signal: if the ball was last seen in the lower fraction
# of the frame (i.e. fell to the floor) AND has been lost for at least the
# grace period, that's a visual drop confirmation independent of cadence.
GROUND_ZONE_Y_FRAC  = 0.78      # bottom 22% of frame = "ground zone"
GROUND_LOST_GRACE_S = 0.4

# --- HSV on-the-fly calibration -----------------------------------------
HSV_CALIB_PATH      = "hsv_calibration.json"
HSV_SAMPLE_BOX_PX   = 60        # side of the center crosshair sampling box
HSV_MARGIN          = np.array([4, 30, 35], dtype=np.int16)  # H, S, V — tight enough to exclude skin
# Caps on the *low* bounds a sample may produce.  A ball sampled off a bright
# specular glint (or in much brighter light than it's played in) yields a high
# S/V floor that then matches nothing in normal light — the V>=220 trap that
# blanked the mask in the field.  Never let the sampled S/V floor rise so high
# the ball's own body is excluded; the hue floor is left alone.
HSV_SAMPLE_SFLOOR_MAX = 110     # S floor is never sampled above this
HSV_SAMPLE_VFLOOR_MAX = 90      # V floor is never sampled above this

# -------------------------------------------------------------------------


class OneEuroFilter:
    """Adaptive low-pass filter from Casiez, Roussel, Vogel (2012).
    Heavy smoothing when input is slow, light smoothing when input is fast.

    Usage:
        f = OneEuroFilter(fc_min=1.0, beta=0.5)
        smoothed = f.filter(raw_value, t_seconds)
        f.reset()  # call when tracking is lost so the next sample is fresh
    """
    def __init__(self, fc_min=1.0, beta=0.5):
        self.fc_min = fc_min
        self.beta   = beta
        self.x_prev  = None
        self.dx_prev = 0.0
        self.t_prev  = None

    def reset(self):
        self.x_prev  = None
        self.dx_prev = 0.0
        self.t_prev  = None

    @staticmethod
    def _alpha(fc, dt):
        tau = 1.0 / (2.0 * math.pi * fc)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 1e-6)
        # Filter the derivative
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.fc_min, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        # Adapt cutoff to filtered velocity
        fc = self.fc_min + self.beta * abs(dx_hat)
        a_x = self._alpha(fc, dt)
        x_hat = a_x * x + (1.0 - a_x) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat


# --- HSV calibration: sample-from-frame on key press --------------------

def load_hsv_calibration(path=HSV_CALIB_PATH):
    """Return (low_a, high_a) or None if no calibration file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        lo = np.array(d["low_a"], dtype=np.uint8)
        hi = np.array(d["high_a"], dtype=np.uint8)
        return lo, hi
    except Exception:
        return None


def save_hsv_calibration(low_a, high_a, path=HSV_CALIB_PATH):
    try:
        with open(path, "w") as f:
            json.dump({"low_a": low_a.tolist(),
                       "high_a": high_a.tolist()}, f)
        return True
    except Exception:
        return False


def sample_hsv_from(frame, best=None, box_px=HSV_SAMPLE_BOX_PX,
                    margin=HSV_MARGIN):
    """Sample HSV inside the detected ball (preferred), or in a centered box
    if no detection. Returns (low_a, high_a) computed from the 5–95th
    percentile of the sampled pixels, expanded by `margin`."""
    h, w = frame.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    if best is not None:
        cx, cy, r = int(best[0]), int(best[1]), max(int(best[2] * 0.7), 4)
        cv2.circle(roi_mask, (cx, cy), r, 255, -1)
    else:
        b = box_px // 2
        cv2.rectangle(roi_mask,
                      (w // 2 - b, h // 2 - b),
                      (w // 2 + b, h // 2 + b), 255, -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    samples = hsv[roi_mask > 0]
    if samples.shape[0] < 100:
        return None
    p_low  = np.percentile(samples, 5,  axis=0).astype(np.int16)
    p_high = np.percentile(samples, 95, axis=0).astype(np.int16)
    lo = np.clip(p_low  - margin, [0, 0, 0],     [179, 255, 255]).astype(np.uint8)
    hi = np.clip(p_high + margin, [0, 0, 0],     [179, 255, 255]).astype(np.uint8)
    # Glint guard: cap the S/V floors so a bright/specular sample can't exclude
    # the ball body in normal light (kills the V>=220 mask-blanking trap).
    lo[1] = min(int(lo[1]), HSV_SAMPLE_SFLOOR_MAX)
    lo[2] = min(int(lo[2]), HSV_SAMPLE_VFLOOR_MAX)
    return lo, hi


# --- Camera (same multi-backend logic as tracker.py) ---------------------

class ThreadedDisplay:
    """cv2.imshow + cv2.waitKey on a background thread.  Main thread's
    show() is non-blocking — it just stores the latest BGR frame per
    window and notifies the display thread.  The display thread pulls
    the snapshot and does the (relatively slow) X/Wayland render.

    Keyboard events are captured by the display thread's waitKey() and
    exposed via get_key()."""

    def __init__(self):
        self._latest    = {}        # title -> latest BGR frame
        self._stop      = False
        self._cond      = threading.Condition()
        self._last_key  = -1
        self._thread    = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def show(self, title, frame):
        with self._cond:
            self._latest[title] = frame
            self._cond.notify_all()

    def _loop(self):
        while not self._stop:
            with self._cond:
                if not self._latest:
                    self._cond.wait(timeout=0.1)
                snap = list(self._latest.items())
                self._latest.clear()
            for title, frame in snap:
                try:
                    cv2.imshow(title, frame)
                except Exception:
                    pass
            try:
                k = cv2.waitKey(1) & 0xFF
            except Exception:
                k = 255
            if k != 255:
                self._last_key = k

    def get_key(self):
        k = self._last_key
        self._last_key = -1
        return k

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=1.0)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _fit_circle_algebraic(pts):
    """Algebraic (Kasa) least-squares circle fit.
    pts: Nx2 array of (x, y) points.
    Returns (cx, cy, r, median_residual_px) or None if the fit is degenerate.

    Linearizes (x-cx)^2 + (y-cy)^2 = r^2 as
        2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2) = x^2 + y^2
    and solves the resulting linear system in (cx, cy, c) where
    c = r^2 - cx^2 - cy^2.  Fast (one np.linalg.lstsq), robust enough
    for "points on a circle plus small noise" — which is our case
    when fragments come from real ball edges."""
    if len(pts) < 3:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c + cx * cx + cy * cy
    if r2 <= 0.0:
        return None
    r = math.sqrt(r2)
    dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return (cx, cy, r, float(np.median(np.abs(dists - r))))


def _try_fragment_circle_fit(contours, anchor):
    """Reconstruct ball position from edge points scattered across one or
    more contours (occluded ball geometries).

    Algorithm:
      1. For every contour, vectorized-filter its points to keep ONLY
         those whose distance from the anchor lies in the expected edge
         band [FRAG_EDGE_INNER_R, FRAG_EDGE_OUTER_R] * anchor.r.  This
         drops finger-chord points (inside the ball), interior speckle,
         and far-away yellow blobs in one shot.
      2. Fit a circle to the surviving edge-band points (Kasa).
      3. Accept only if (a) fitted center is near the anchor, (b) fitted
         radius matches the anchor, and (c) points actually lie on the
         fitted circle (low median residual).

    Returns (cx, cy, r) or None.  Random yellow specks fail (c)."""
    ax, ay, ar = anchor[0], anchor[1], anchor[2]
    if ar <= 0:
        return None
    inner_r2 = (ar * FRAG_EDGE_INNER_R) ** 2
    outer_r2 = (ar * FRAG_EDGE_OUTER_R) ** 2
    anchor_pt = np.array([ax, ay], dtype=np.float32)
    edge_chunks = []
    for c in contours:
        if cv2.contourArea(c) < FRAGMENT_MIN_AREA:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        diff = pts - anchor_pt
        dist_sq = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
        in_band = (dist_sq >= inner_r2) & (dist_sq <= outer_r2)
        if in_band.any():
            edge_chunks.append(pts[in_band])
    if not edge_chunks:
        return None
    edge_pts = np.concatenate(edge_chunks, axis=0)
    if edge_pts.shape[0] < FRAGMENT_MIN_POINTS:
        return None
    fit = _fit_circle_algebraic(edge_pts)
    if fit is None:
        return None
    fcx, fcy, fr, fres = fit
    # Validate: fitted circle must look like the ball, not just any circle.
    if math.hypot(fcx - ax, fcy - ay) > ar * FRAGMENT_CENTER_TOL:
        return None    # center drifted too far from anchor
    if abs(fr - ar) / ar > FRAGMENT_RADIUS_TOL:
        return None    # wrong size
    if fres > FRAGMENT_RESIDUAL_PX:
        return None    # points don't actually lie on a common circle — specks
    return (float(fcx), float(fcy), float(fr))


def _color_blob_centroid(mask, ax_m, ay_m, ar_m):
    """Centroid of the coloured (non-zero) mask pixels inside a window around
    (ax_m, ay_m) — all in mask/detection-resolution coords.  Returns
    (cx_m, cy_m) or None if there isn't enough coloured mass.

    Loose-lock's last resort: a fast/blurred ball smears into a non-round patch
    of the right colour that fails every shape gate.  Tracking the colour
    centroid in a tight window around the anchor keeps the lock with no shape
    requirement; the window + fill threshold keep it from grabbing stray colour."""
    half = max(ar_m * BLOB_WIN_R, BLOB_WIN_MIN_PX)
    mh, mw = mask.shape[:2]
    x0 = max(int(ax_m - half), 0); x1 = min(int(ax_m + half), mw)
    y0 = max(int(ay_m - half), 0); y1 = min(int(ay_m + half), mh)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = mask[y0:y1, x0:x1]
    need = max(BLOB_MIN_FILL * math.pi * ar_m * ar_m, float(FRAGMENT_MIN_AREA))
    if cv2.countNonZero(roi) < need:
        return None
    m = cv2.moments(roi, binaryImage=True)
    if m["m00"] <= 0:
        return None
    return (x0 + m["m10"] / m["m00"], y0 + m["m01"] / m["m00"])


class MaskContourWorker:
    """Background pipeline stage that does the heavy CV work — HSV mask,
    morphology, contour finding, ball-shape scoring — on its own thread.
    Main thread submits frames; it gets back (frame, mask, best) tuples
    where `best` is (cx, cy, r) of the most ball-shaped contour, or None.

    Always-latest semantics: only the freshest input matters, only the
    freshest output is returned.  Frames in between get dropped, which
    is exactly what we want for real-time tracking."""

    def __init__(self, kernel_open, kernel_close,
                 min_radius_px, max_radius_px, min_circularity,
                 min_radius_px_strict, min_circularity_strict):
        self.kernel_open     = kernel_open
        self.kernel_close    = kernel_close
        # Tracking-mode (have an anchor) — permissive.
        self.min_radius_px   = min_radius_px
        self.max_radius_px   = max_radius_px
        self.min_circularity = min_circularity
        # Re-acquire-mode (no anchor) — strict, only ball-shaped wins.
        self.min_radius_px_strict   = min_radius_px_strict
        self.min_circularity_strict = min_circularity_strict
        # Absolute size floor applied in every mode (tunable at runtime).
        self.min_ball_radius_px     = MIN_BALL_RADIUS_PX
        # Relaxed circularity for ghost-anchor re-acquire (tunable at runtime).
        self.ghost_reacquire_circ   = GHOST_REACQUIRE_CIRC
        # Loose-lock mode (set at runtime from the web UI): drop the shape gate
        # and enable the colour-blob fallback so blur keeps the lock.
        self.loose_lock             = False
        self._in_q   = queue.Queue(maxsize=1)
        self._out_q  = queue.Queue(maxsize=1)
        self._stop   = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame, frame_idx, t, anchor=None, ghost_anchor=None,
               max_jump_px=None):
        """anchor: (cx, cy, r) of recent detection — when set, enables
            tracking mode (loose gates + size lock + proximity bonus).
        ghost_anchor: (cx, cy) of last-known ball position — used in
            re-acquire mode (strict gates) as a proximity prior so the
            ball wins over distant specs when it reappears.
        max_jump_px: speed-aware hard distance gate for tracking mode; None
            falls back to the flat TRACK_HARD_PX."""
        # Drop the previous unprocessed input; keep only the latest.
        try:
            self._in_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._in_q.put_nowait((frame, frame_idx, t, anchor, ghost_anchor,
                                   max_jump_px))
        except queue.Full:
            pass

    def get_latest(self):
        """Drain output queue, return only the most recent result."""
        latest = None
        while True:
            try:
                latest = self._out_q.get_nowait()
            except queue.Empty:
                break
        return latest

    def _loop(self):
        while not self._stop:
            try:
                (frame, frame_idx, t, anchor, ghost_anchor,
                 max_jump_px) = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            # Per-frame hard-distance gate (speed-aware; falls back to the flat
            # TRACK_HARD_PX when the caller didn't supply one).
            hard_px = max_jump_px if max_jump_px is not None else TRACK_HARD_PX
            # Gate modes, most→least permissive by how strong our prior is:
            #   anchor  -> loose (tracking through partial occlusion)
            #   ghost   -> strict radius (reject specks) + RELAXED circularity,
            #              since the proximity prior guards spatially and a
            #              reappearing ball is often occluded/blurred (low circ)
            #   neither -> fully strict (cold-start re-acquire, no prior)
            if anchor is not None:
                min_r  = self.min_radius_px
                min_c  = self.min_circularity
            elif ghost_anchor is not None:
                min_r  = self.min_radius_px_strict
                min_c  = self.ghost_reacquire_circ
            else:
                min_r  = self.min_radius_px_strict
                min_c  = self.min_circularity_strict
            # Loose-lock: with a spatial prior (anchor/ghost) drop the gates so
            # blurry/partial shapes qualify.  Cold-start stays selective (no
            # prior -> low circ would latch onto any coloured noise).
            if self.loose_lock and (anchor is not None or ghost_anchor is not None):
                min_c = min(min_c, LOOSE_LOCK_CIRC)
                min_r = min(min_r, self.min_ball_radius_px)
            # Absolute floor — never accept a ball smaller than this, in any
            # mode.  Kills tiny-speck lock-on and anchor ratcheting.
            min_r = max(min_r, self.min_ball_radius_px)
            min_area = math.pi * min_r * min_r * 0.5
            try:
                # Run the heavy pixel ops at ~640 wide regardless of capture
                # resolution, then scale the contours back to full res.  This
                # keeps detection cost ~constant (so 1280x720 stays fast) while
                # the display frame stays full-resolution and sharp.
                fh, fw = frame.shape[:2]
                ms = 640.0 / fw if fw > 640 else 1.0
                inv = 1.0 / ms                  # mask-coords -> full-res scale
                src = (cv2.resize(frame, (int(fw * ms), int(fh * ms)),
                                  interpolation=cv2.INTER_LINEAR)
                       if ms < 1.0 else frame)
                # --- HSV mask (reads module-level bounds; updated by 'h'). ---
                blurred = cv2.GaussianBlur(src, (5, 5), 0)
                hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                mask_a = cv2.inRange(hsv, HSV_LOW_A, HSV_HIGH_A)
                mask_b = cv2.inRange(hsv, HSV_LOW_B, HSV_HIGH_B)
                mask = cv2.bitwise_or(mask_a, mask_b)
                mask = cv2.medianBlur(mask, 5)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel_open)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

                # --- Best ball-shaped contour (scaled back to full res). -----
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if ms < 1.0 and contours:
                    contours = [np.round(c.astype(np.float32) * inv).astype(np.int32)
                                for c in contours]
                best, best_score = None, 0.0
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < min_area:
                        continue
                    perim = cv2.arcLength(c, True)
                    if perim < 1.0:
                        continue
                    circ = 4.0 * math.pi * area / (perim * perim)
                    if circ < min_c:
                        continue
                    (cx_c, cy_c), rr = cv2.minEnclosingCircle(c)
                    if not (min_r <= rr <= self.max_radius_px):
                        continue
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx_m = M["m10"] / M["m00"]
                        cy_m = M["m01"] / M["m00"]
                    else:
                        cx_m, cy_m = cx_c, cy_c
                    # Score is mode-dependent.  When we have an anchor (or
                    # a ghost anchor) the size gate has already filtered to
                    # plausible ball-sized blobs, so AREA carries no useful
                    # signal — it just helps big body chunks beat the actual
                    # ball.  Use roundness + proximity only.  Only cold-start
                    # uses area (to pick the obvious "biggest round yellow
                    # thing" with no spatial prior).
                    if anchor is not None:
                        # Tracking mode — strict anchor (size + position lock).
                        dx = cx_m - anchor[0]
                        dy = cy_m - anchor[1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        # HARD REJECT — outside the (speed-aware) radius the
                        # ball could plausibly have moved to since the last real
                        # fix.  hard_px grows with measured speed so a fast throw
                        # isn't rejected for "jumping too far" (esp. low fps).
                        if dist > hard_px:
                            continue
                        # Asymmetric size tolerance: shrinking is OK
                        # (occlusion), growing past TRACK_GROW_TOL is not.
                        if anchor[2] > 0:
                            delta = (rr - anchor[2]) / anchor[2]
                            if delta >  TRACK_GROW_TOL:
                                continue
                            if delta < -TRACK_SHRINK_TOL:
                                continue
                        prox = math.exp(-dist / TRACK_SEARCH_PX)
                        # circ^2 (roundness dominates) × proximity bonus.
                        # No area term — size gate already constrains size.
                        score = (circ * circ) * (1.0 + TRACK_BONUS * prox)
                    elif ghost_anchor is not None:
                        # Re-acquire mode WITH spatial memory — strict gates
                        # filter min size/roundness; score by roundness +
                        # proximity to last-known position.  No area term:
                        # body chunks near the last position shouldn't win
                        # just for being bigger than the actual ball.
                        dx = cx_m - ghost_anchor[0]
                        dy = cy_m - ghost_anchor[1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        prox = math.exp(-dist / GHOST_ANCHOR_RANGE_PX)
                        score = (circ * circ) * (1.0 + GHOST_ANCHOR_BONUS * prox)
                    else:
                        # Cold start — no spatial info, area helps find the
                        # obvious "big round yellow thing" against noise.
                        score = (circ * circ) * math.sqrt(area)
                    if score > best_score:
                        best_score = score
                        best = (float(cx_m), float(cy_m), float(rr))

                # --- Heavy-occlusion fallback (lacrosse mesh / hand). ----
                # If we have an anchor (we know where the ball was) but no
                # single contour passed as a plausible whole-ball detection
                # — OR the best we got is much smaller than the anchor
                # (heavy occlusion exposing only a fragment) — try to
                # reconstruct the ball geometrically from multiple visible
                # fragments lying on a common circle.
                if anchor is not None and anchor[2] > 0:
                    need_fit = (
                        best is None
                        or best[2] < anchor[2] * FRAGMENT_FIT_TRIGGER_R
                    )
                    if need_fit:
                        synth = _try_fragment_circle_fit(contours, anchor)
                        if synth is not None:
                            best = synth

                # --- Colour-blob fallback (loose-lock only). -------------
                # Nothing passed even the relaxed gates and geometric fit, but
                # in loose-lock we still track a blurred smear of the right
                # colour by its centroid in a window around the anchor.  Keeps
                # the lock through fast motion; off by default.
                if (self.loose_lock and best is None
                        and anchor is not None and anchor[2] > 0):
                    c = _color_blob_centroid(mask, anchor[0] * ms,
                                             anchor[1] * ms, anchor[2] * ms)
                    if c is not None:
                        best = (float(c[0] * inv), float(c[1] * inv),
                                float(anchor[2]))
            except Exception:
                continue
            # Replace any stale output with the new result.
            try:
                self._out_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._out_q.put_nowait((frame, frame_idx, t, mask, best))
            except queue.Full:
                pass

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)


class ThreadedCapture:
    """Background thread that continuously calls the underlying read() and
    keeps the *latest* frame available.  Main thread reads via read()
    which blocks only until a new (not yet consumed) frame is ready.
    Lets the capture I/O run on its own core while the main loop's
    processing runs on another."""
    def __init__(self, raw_read):
        self._raw_read = raw_read
        self._latest   = None
        self._seq      = 0
        self._consumed = 0
        self._stop     = False
        self._cond     = threading.Condition()
        self._cap_count   = 0          # total frames captured
        self._cap_started = time.time()
        self._thread   = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self._raw_read()
            if ok and frame is not None:
                with self._cond:
                    self._latest = frame
                    self._seq   += 1
                    self._cap_count += 1
                    self._cond.notify_all()
            else:
                time.sleep(0.001)

    def capture_fps(self):
        dt = max(time.time() - self._cap_started, 1e-6)
        return self._cap_count / dt

    def read(self):
        with self._cond:
            while self._seq <= self._consumed and not self._stop:
                self._cond.wait(timeout=0.5)
            if self._stop or self._latest is None:
                return False, None
            self._consumed = self._seq
            return True, self._latest

    def release(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=1.0)


def _try_open_gstreamer(w, h, target_fps):
    """GStreamer pipeline with hardware MJPEG decode (Rockchip MPP).
    Offloads MJPEG->BGR off the CPU, freeing cores for detection — this is
    what makes 1280x720@60 possible. Returns (cap, label) or (None, err)."""
    if not sys.platform.startswith("linux"):
        return None, "not linux"
    import subprocess
    decoder = None
    for el in ("mppjpegdec", "v4l2jpegdec", "jpegdec"):
        try:
            res = subprocess.run(["gst-inspect-1.0", el],
                                 capture_output=True, timeout=2, text=True)
            if res.returncode == 0 and res.stdout.strip():
                decoder = el
                break
        except Exception:
            continue
    if decoder is None:
        return None, "no jpeg decoder element found"
    pipeline = (
        f"v4l2src device=/dev/video0 io-mode=4 ! "
        f"image/jpeg,width={w},height={h},framerate={target_fps}/1 ! "
        f"{decoder} ! videoconvert n-threads=4 ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=true sync=false max-buffers=1"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        return None, f"pipeline failed to open ({decoder})"
    time.sleep(0.5)
    for _ in range(10):
        ok, _f = cap.read()
        if ok:
            return cap, f"GST:{decoder}"
        time.sleep(0.1)
    cap.release()
    return None, f"pipeline opened but no frames ({decoder})"


def open_camera(w, h, return_cap=False):
    """Plain V4L2 / DSHOW / MSMF camera open.  No GStreamer, no exposure
    twiddling — the camera's own auto-exposure / auto-WB stays on.

    return_cap=True also returns the underlying cv2.VideoCapture as a 4th
    element.  Prefers GStreamer hardware MJPEG decode (Rockchip mppjpegdec)
    on Linux for high fps; falls back to direct V4L2 (CPU decode)."""
    # 1) GStreamer hardware-decode path (Linux) — the high-fps route.
    gcap, glabel = _try_open_gstreamer(w, h, TARGET_FPS)
    if gcap is not None:
        def gread():
            for _ in range(3):
                ok, fr = gcap.read()
                if ok:
                    return True, fr
                time.sleep(0.005)
            return False, None
        print(f"camera: {glabel} {w}x{h}@{TARGET_FPS}", flush=True)
        if return_cap:
            return gread, gcap.release, glabel, gcap
        return gread, gcap.release, glabel

    # 2) Direct V4L2 / DSHOW / MSMF fallback (CPU MJPEG decode).
    backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF),
                ("ANY",   cv2.CAP_ANY)] \
        if sys.platform.startswith("win") else \
        [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]
    last_err = None

    def try_open(idx, backend, set_res):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            return None
        if set_res:
            # Best-effort.  Camera ignores anything it can't deliver and
            # falls back to its native size / fourcc; we resize in the
            # main loop if needed.
            try:
                # Request MJPEG first — most USB webcams deliver much higher
                # fps + sharper frames in MJPEG than the default YUYV.
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                # Smallest internal queue — always give us the freshest frame.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        time.sleep(0.8)
        for _ in range(30):
            ok, _f = cap.read()
            if ok:
                return cap
            time.sleep(0.1)
        cap.release()
        return None

    for attempt in range(4):
        for name, backend in backends:
            for idx in range(3):
                # First try with resolution hint, then without (Linux fallback).
                cap = try_open(idx, backend, set_res=True)
                if cap is None:
                    cap = try_open(idx, backend, set_res=False)
                if cap is None:
                    last_err = f"{name}:{idx} opened but no frames"
                    continue
                # Report what the camera actually negotiated.
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc = "".join(
                    chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                afps = cap.get(cv2.CAP_PROP_FPS)
                print(f"camera negotiated: {aw}x{ah} @ {afps:.1f}fps "
                      f"fourcc={fourcc!r}")
                def make_read(c):
                    def _read():
                        for _ in range(3):
                            ok, fr = c.read()
                            if ok:
                                return True, fr
                            time.sleep(0.02)
                        return False, None
                    return _read
                if return_cap:
                    return make_read(cap), cap.release, f"{name}:{idx}", cap
                return make_read(cap), cap.release, f"{name}:{idx}"
        if attempt < 3:
            time.sleep(1.5)
    raise RuntimeError(f"No camera available. {last_err or ''}")


# --- Trajectory analysis -------------------------------------------------

def r2_score(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0


def _recent_velocity(trail, span_s):
    """Estimate the ball's recent image-plane velocity (px/s) from the trail.
    Uses the latest point and the most recent point at least `span_s` older so
    frame-to-frame noise at high fps doesn't dominate.  Returns (vx, vy), or
    (0.0, 0.0) when there isn't enough history."""
    if len(trail) < 2:
        return 0.0, 0.0
    last = trail[-1]
    ref = trail[0]
    for p in reversed(trail):
        if last[0] - p[0] >= span_s:
            ref = p
            break
    dt = last[0] - ref[0]
    if dt < 1e-6:
        return 0.0, 0.0
    return (last[1] - ref[1]) / dt, (last[2] - ref[2]) / dt


def coasted_anchor(last_seen_pos, last_seen_at, last_known_r, trail, now,
                   frame_w, frame_h):
    """Build the worker's (anchor, ghost_anchor, max_jump_px) for this frame,
    sliding the last-known position along the ball's measured velocity so the
    search region follows a fast/blurred ball through frames where detection
    momentarily drops (throw survival).  Constant-velocity is only good
    briefly, so the extrapolation is capped at COAST_MAX_S.

      - within TRACK_LOST_GAP_S + COAST_EXTRA_S -> strict anchor (loose gates)
      - else within GHOST_ANCHOR_GAP_S          -> ghost anchor (strict gates)
      - else                                    -> cold start (both None)

    max_jump_px (the tracking-mode hard distance gate) grows with measured
    speed so a genuinely fast ball isn't rejected for jumping too far — which
    matters when fps dips and the per-frame displacement is large."""
    if last_seen_pos is None:
        return None, None, TRACK_HARD_PX
    age = now - last_seen_at
    vx, vy = _recent_velocity(trail, COAST_VEL_SPAN_S)
    coast_dt = min(age, COAST_MAX_S)
    px = float(np.clip(last_seen_pos[0] + vx * coast_dt, 0, frame_w - 1))
    py = float(np.clip(last_seen_pos[1] + vy * coast_dt, 0, frame_h - 1))
    speed = math.hypot(vx, vy)
    max_jump = float(np.clip(TRACK_JUMP_K * speed * max(age, 1e-3),
                             TRACK_HARD_PX, TRACK_JUMP_MAX_PX))
    if age < TRACK_LOST_GAP_S + COAST_EXTRA_S and last_known_r is not None:
        return (px, py, last_known_r), None, max_jump
    if age < GHOST_ANCHOR_GAP_S:
        return None, (px, py), max_jump
    return None, None, TRACK_HARD_PX


def _fit_window(window, wall_dir):
    """Fit one suffix; return result dict only if the motion is outbound
    AND in flight (not noise/drift).  wall_dir: +1 right, -1 left."""
    t = np.array([p[0] for p in window], dtype=np.float64)
    x = np.array([p[1] for p in window], dtype=np.float64)
    y = np.array([p[2] for p in window], dtype=np.float64)
    r = np.array([p[3] for p in window], dtype=np.float64)
    t -= t[0]
    if t[-1] < MIN_DURATION_S:
        return None

    # Distance-scale all pixel thresholds by median ball radius.
    r_med = float(np.median(r))
    scale = max(r_med / REF_RADIUS_PX, MIN_SCALE)
    eff_disp  = MIN_TOTAL_DISP_PX     * scale
    eff_a     = MIN_A_PX_S2           * scale
    eff_v     = max(MIN_PEAK_SPEED_PX * scale, MIN_PEAK_SPEED_FLOOR_PX)
    eff_vx    = MIN_OUTBOUND_VX_PX    * scale

    if float(np.hypot(x[-1] - x[0], y[-1] - y[0])) < eff_disp:
        return None

    dts = np.diff(t)
    dxs = np.diff(x)
    dys = np.diff(y)
    safe_dt = np.where(dts > 1e-6, dts, 1e-6)
    speeds = np.hypot(dxs, dys) / safe_dt
    peak_speed = float(np.max(speeds)) if speeds.size else 0.0
    if peak_speed < eff_v:
        return None

    ay, by, cy = np.polyfit(t, y, 2)
    mx, kx     = np.polyfit(t, x, 1)

    # Outbound gate — the wall-ball-specific filter.
    if wall_dir * mx < eff_vx:
        return None

    r2y = r2_score(y, np.polyval([ay, by, cy], t))
    r2x = r2_score(x, np.polyval([mx, kx], t))
    thrown = (ay > eff_a) and (r2y > MIN_R2_Y) and (r2x > MIN_R2_X)

    apex_contained = False
    if ay > 1e-6:
        t_apex = -by / (2.0 * ay)
        apex_contained = (APEX_EDGE_MARGIN_S < t_apex <
                          t[-1] - APEX_EDGE_MARGIN_S)

    return {
        "thrown": bool(thrown),
        "ay": float(ay), "r2y": float(r2y), "r2x": float(r2x),
        "vx": float(mx),
        "peak_speed": peak_speed,
        "duration": float(t[-1]),
        "fit_y": (float(ay), float(by), float(cy)),
        "fit_x": (float(mx), float(kx)),
        "n": len(window),
        "apex_contained": bool(apex_contained),
        "r_med": r_med,
        "scale": scale,
    }


def _check_burst(points, wall_dir):
    """Lacrosse-stick path: short burst of fast outbound motion, no parabola
    required. Filters out stick-mesh flicker via the outbound-fraction check."""
    n = len(points)
    if n < MIN_BURST_POINTS:
        return None

    t = np.array([p[0] for p in points], dtype=np.float64)
    x = np.array([p[1] for p in points], dtype=np.float64)
    y = np.array([p[2] for p in points], dtype=np.float64)
    r = np.array([p[3] for p in points], dtype=np.float64)
    t -= t[0]
    if t[-1] < MIN_BURST_DURATION_S:
        return None

    r_med = float(np.median(r))
    scale = max(r_med / REF_RADIUS_PX, MIN_SCALE)

    dx = float(x[-1] - x[0])
    if wall_dir * dx < MIN_BURST_DX_PX * scale:
        return None

    dts = np.diff(t)
    dxs = np.diff(x)
    dys = np.diff(y)
    safe_dt = np.where(dts > 1e-6, dts, 1e-6)
    speeds = np.hypot(dxs, dys) / safe_dt
    peak_speed = float(np.max(speeds)) if speeds.size else 0.0
    if peak_speed < max(MIN_BURST_SPEED_PX * scale, MIN_PEAK_SPEED_FLOOR_PX):
        return None

    # Frame-to-frame x-delta sign — rejects stick-mesh flicker (random jumps).
    x_signs = dxs * wall_dir
    if x_signs.size:
        outbound = float(np.sum(x_signs > 0)) / float(x_signs.size)
        if outbound < MIN_OUTBOUND_FRAC:
            return None

    return {
        "thrown": True,
        "burst":  True,
        "ay": 0.0, "r2y": 0.0, "r2x": 0.0,
        "vx": float(dx / t[-1]),
        "peak_speed": peak_speed,
        "duration": float(t[-1]),
        "fit_y": (0.0, 0.0, float(y[0])),
        "fit_x": (float(dx / t[-1]), float(x[0])),
        "n": n,
        "apex_contained": False,
        "r_med": r_med,
        "scale": scale,
    }


def analyse_trajectory(points, wall_dir):
    """Try the parabolic path first (suffix windows, biggest first); fall back
    to the burst detector when no clean parabolic fit exists."""
    if len(points) > MAX_FIT_PTS:        # only the recent tail matters for a throw
        points = points[-MAX_FIT_PTS:]
    n = len(points)
    if n < MIN_BURST_POINTS:
        return None

    last_valid = None
    if n >= MIN_POINTS_FIT:
        for size in range(n, MIN_POINTS_FIT - 1, -1):
            result = _fit_window(points[-size:], wall_dir)
            if result is None:
                continue
            if result["thrown"]:
                return result
            if last_valid is None:
                last_valid = result

    burst = _check_burst(points, wall_dir)
    if burst is not None:
        return burst
    return last_valid


# --- Wall-ball-specific components ---------------------------------------

class CadenceTracker:
    """Rolling throws-per-minute over a time window."""
    def __init__(self, window_s=CADENCE_WINDOW_S):
        self.window_s = window_s
        self.throws = collections.deque()

    def add(self, t):
        self.throws.append(t)

    def reset(self):
        self.throws.clear()

    def rate(self, now):
        cutoff = now - self.window_s
        while self.throws and self.throws[0] < cutoff:
            self.throws.popleft()
        if not self.throws:
            return 0.0
        elapsed = max(now - self.throws[0], 1e-6)
        return len(self.throws) * 60.0 / min(elapsed, self.window_s)

    def last_throw_ago(self, now):
        return None if not self.throws else (now - self.throws[-1])

    def median_interval(self, lookback=DROP_LOOKBACK):
        """Median of the last `lookback` inter-rep intervals, or None if
        we don't have enough reps yet."""
        if len(self.throws) < DROP_MIN_REPS:
            return None
        throws = list(self.throws)
        intervals = [throws[i] - throws[i-1] for i in range(1, len(throws))]
        intervals = intervals[-lookback:]
        if not intervals:
            return None
        return float(np.median(intervals))

    def gap_is_drop(self, now,
                    ratio=DROP_CADENCE_RATIO,
                    min_gap_s=DROP_MIN_GAP_S):
        """If the current gap since the last rep looks like a drop relative
        to recent cadence, return (gap, median, threshold). Else None."""
        med = self.median_interval()
        if med is None or med < 0.05:
            return None
        threshold = max(min_gap_s, ratio * med)
        last_ago = now - self.throws[-1]
        if last_ago >= threshold:
            return last_ago, med, threshold
        return None


class SessionLogger:
    """CSV log: one row per rep + drop, summary footer on close."""
    def __init__(self, path):
        self.path = path
        self.started_at = time.time()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "wall_time_iso", "session_t", "event", "rep_idx",
            "ay_pxps2", "vx_pxps", "peak_speed_pxps",
            "r2y", "r2x", "n_points", "ball_radius_px",
            "pose_blob",
        ])
        self._file.flush()    # so the header survives a SIGTERM/kill

    def log_throw(self, now, rep_idx, info, pose_blob=""):
        self._writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            f"{now:.3f}", "throw", rep_idx,
            f"{info['ay']:.1f}", f"{info['vx']:.1f}",
            f"{info['peak_speed']:.1f}",
            f"{info['r2y']:.3f}", f"{info['r2x']:.3f}",
            info['n'], f"{info['r_med']:.1f}",
            pose_blob,
        ])
        self._file.flush()

    def log_drop(self, now, since_last):
        self._writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            f"{now:.3f}", "drop", "", "", "", "", "", "", "", "",
            f"gap={since_last:.2f}s",
        ])
        self._file.flush()

    def close(self, summary):
        self._writer.writerow([])
        self._writer.writerow(["# SUMMARY"])
        for k, v in summary.items():
            self._writer.writerow([f"# {k}", v])
        self._file.close()


class PoseEstimator:
    """No-op default.  Drop in a real implementation later:

        class MediaPipePose(PoseEstimator):
            enabled = True
            def __init__(self):
                import mediapipe as mp
                self.solver = mp.solutions.pose.Pose(
                    model_complexity=1, min_detection_confidence=0.5)
            def __call__(self, frame):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.solver.process(rgb)
                if not res.pose_landmarks:
                    return None
                lm = res.pose_landmarks.landmark
                # compute angles, release height, etc.
                return {
                    "shoulder_angle_deg": ...,
                    "elbow_angle_deg":    ...,
                    "release_height_px":  ...,
                    "landmarks_xy":       [(p.x, p.y) for p in lm],
                }
            def annotate(self, frame, info):
                # draw landmarks on frame (in-place)
                ...

    The main loop calls self(frame) every frame; the returned dict (if any)
    is stamped into the CSV alongside each confirmed rep so you can correlate
    form features with cadence later.
    """
    enabled = False

    def __call__(self, frame):
        return None

    def annotate(self, frame, info):
        pass

    @staticmethod
    def serialize(info):
        """Turn the pose dict into a single CSV cell (JSON-ish, no commas).
        Customise to taste once a real estimator is wired up."""
        if info is None:
            return ""
        parts = []
        for k in ("shoulder_angle_deg", "elbow_angle_deg",
                  "release_height_px"):
            if k in info:
                parts.append(f"{k}={info[k]:.1f}")
        return ";".join(parts)


# --- Main loop -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-display", action="store_true",
                    help="headless mode (Pi with no monitor)")
    ap.add_argument("--video", type=str, default=None,
                    help="read from a file instead of the camera")
    ap.add_argument("--wall-side", choices=["right", "left"],
                    default="right",
                    help="which side of the frame the wall is on")
    ap.add_argument("--log-dir", type=str, default="sessions",
                    help="directory for the per-session CSV log")
    args = ap.parse_args()

    wall_dir = +1 if args.wall_side == "right" else -1

    # Cap OpenCV's internal thread pool so it doesn't fight with the
    # GStreamer pipeline for cores.
    try:
        cv2.setNumThreads(CV_INTERNAL_THREADS)
    except Exception:
        pass

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {args.video}")
        read, release, src = cap.read, cap.release, "file"
        threaded = None
    else:
        raw_read, raw_release, src = open_camera(FRAME_W, FRAME_H)
        # Decouple capture I/O from processing — the read thread runs on
        # its own core while the main loop runs on another.
        threaded = ThreadedCapture(raw_read)
        read = threaded.read
        def release():
            threaded.release()
            raw_release()
    print(f"source: {src}  wall: {args.wall_side}")

    log_path = os.path.join(
        args.log_dir,
        f"wallball_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    logger = SessionLogger(log_path)
    print(f"logging to {log_path}")

    cadence = CadenceTracker()
    pose = PoseEstimator()           # <-- swap in a real estimator here later
    display = None if args.no_display else ThreadedDisplay()

    # Snapshot the compiled-in HSV defaults so 'H' can revert to them.
    HSV_LOW_A_DEFAULT  = HSV_LOW_A.copy()
    HSV_HIGH_A_DEFAULT = HSV_HIGH_A.copy()
    # Load saved calibration from a previous run (if any).
    _saved = load_hsv_calibration()
    if _saved is not None:
        HSV_LOW_A[:], HSV_HIGH_A[:] = _saved
        print(f"loaded HSV calibration: low={HSV_LOW_A.tolist()} "
              f"high={HSV_HIGH_A.tolist()}")

    trail = collections.deque(maxlen=TRAIL_LEN)
    # Bigger kernels for 1280x800 / larger ball: open kills small noise
    # specks; close fills bigger holes inside the ball.
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # Heavy CV runs on its own thread — gives the OS scheduler something
    # real to spread across cores in parallel with the main loop's Python
    # work + the display thread + the GStreamer threads.
    cv_worker = MaskContourWorker(kernel_open, kernel_close,
                                  MIN_RADIUS_PX, MAX_RADIUS_PX,
                                  MIN_CIRCULARITY,
                                  MIN_RADIUS_PX_STRICT,
                                  MIN_CIRCULARITY_STRICT)
    fps_window = collections.deque(maxlen=60)   # last 60 frame timestamps
    frame_idx  = 0
    last_fps_print = time.time()
    # One-Euro per-axis filters that adapt smoothing to motion speed.
    filter_cx = OneEuroFilter(fc_min=ONEEURO_FC_MIN,   beta=ONEEURO_BETA)
    filter_cy = OneEuroFilter(fc_min=ONEEURO_FC_MIN,   beta=ONEEURO_BETA)
    filter_r  = OneEuroFilter(fc_min=ONEEURO_R_FC_MIN, beta=ONEEURO_R_BETA)

    start = time.time()
    rep_count        = 0
    drop_count       = 0
    last_seen_at     = 0.0
    last_seen_pos    = None   # (cx, cy) of most recent ball detection
    last_known_r     = None   # radius from most recent detection
    pending_info     = None
    pending_set_at   = 0.0
    held_info        = None
    label_hold_until = 0.0
    drop_armed       = False   # set True after each rep; clears after a drop fires
    # Transient status message for the HUD (HSV recalibrated, etc).
    hsv_status       = ""
    hsv_status_until = 0.0
    cooldown_until   = 0.0     # suppress new reps until this session time

    try:
        while True:
            # 1. Capture latest frame from the GStreamer pipeline.
            ok, capture_frame = read()
            if not ok:
                break
            if (capture_frame.shape[1] != FRAME_W
                    or capture_frame.shape[0] != FRAME_H):
                capture_frame = cv2.resize(capture_frame, (FRAME_W, FRAME_H))
            capture_t = time.time() - start

            # 2. Submit to the worker (HSV mask + morph + best contour).
            # Two layers of position memory:
            #   - anchor:       strict (within TRACK_LOST_GAP_S) — full
            #                   size+position lock for tight tracking.
            #   - ghost_anchor: soft  (within GHOST_ANCHOR_GAP_S)  — last
            #                   known position only.  Strict gates still
            #                   apply, but blobs near here get a score
            #                   bonus so the ball wins over noise when
            #                   it pops back out of an occluding stick.
            # Coast the anchor along the ball's measured velocity and widen the
            # hard gate with speed (throw survival — see coasted_anchor()).
            anchor, ghost_anchor, max_jump_px = coasted_anchor(
                last_seen_pos, last_seen_at, last_known_r, trail,
                capture_t, FRAME_W, FRAME_H)
            cv_worker.submit(capture_frame, frame_idx, capture_t,
                             anchor, ghost_anchor, max_jump_px)
            frame_idx += 1

            # 3. Pull the freshest worker result.  None on the first few
            #    iterations before the worker has anything done.
            result = cv_worker.get_latest()
            if result is None:
                continue
            frame, _w_idx, now, mask, best = result

            do_draw = (not args.no_display
                       and (frame_idx % DISPLAY_EVERY_N) == 0)
            fps_window.append(now)
            if len(fps_window) >= 2:
                cur_fps = (len(fps_window) - 1) / max(
                    fps_window[-1] - fps_window[0], 1e-6)
            else:
                cur_fps = 0.0

            # Periodic diagnostic: print main-loop and capture-thread fps.
            if (time.time() - last_fps_print) >= 2.0:
                if threaded is not None:
                    cap_fps = threaded.capture_fps()
                else:
                    cap_fps = cur_fps    # no separate capture thread
                print(f"[fps] main={cur_fps:.1f}  capture={cap_fps:.1f}  "
                      f"trail={len(trail)}", flush=True)
                last_fps_print = time.time()

            # --- Pose hook (no-op by default) ----------------------------
            pose_info = pose(frame) if pose.enabled else None
            if pose_info is not None:
                pose.annotate(frame, pose_info)

            # Forget stale trail points.
            while trail and (now - trail[0][0]) > TRAIL_MAX_AGE_S:
                trail.popleft()

            if best is not None:
                cx_raw, cy_raw, rr_raw = best
                # One-Euro adaptive smoothing on every channel.
                cx = filter_cx.filter(cx_raw, now)
                cy = filter_cy.filter(cy_raw, now)
                rr = filter_r .filter(rr_raw, now)
                trail.append((now, cx, cy, rr))
                last_seen_at = now
                last_seen_pos = (cx, cy)
                last_known_r  = rr
                if do_draw:
                    cv2.circle(frame, (int(cx), int(cy)), int(rr),
                               (0, 255, 255), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 3,
                               (0, 0, 255), -1)
            else:
                # Only reset filters after sustained absence — brief 1-2
                # frame detection drops shouldn't break filter continuity.
                if (last_seen_at > 0
                        and (now - last_seen_at) > FILTER_RESET_GAP_S):
                    filter_cx.reset()
                    filter_cy.reset()
                    filter_r.reset()

            # --- Trail polyline (display only) ---------------------------
            if do_draw and len(trail) >= 2:
                pts = np.array(
                    [(int(p[1]), int(p[2])) for p in trail],
                    dtype=np.int32)
                cv2.polylines(frame, [pts], False, (255, 255, 255), 2)

            # --- Throw analysis + release confirmation --------------------
            # Only do the (relatively expensive) suffix-window fit when there
            # might actually be useful work: a fresh ball detection arrived,
            # OR a pending candidate is waiting for its APEX/SUST recheck.
            if best is not None or pending_info is not None:
                info = analyse_trajectory(list(trail), wall_dir)
            else:
                info = None

            if (info and info["thrown"] and pending_info is None
                    and now >= cooldown_until):
                pending_info  = info
                pending_set_at = now

            if pending_info is not None:
                latest   = info if info is not None else pending_info
                lost_for = now - last_seen_at
                pending_age = now - pending_set_at
                path_a = lost_for >= RELEASE_GAP_S
                path_b = (latest.get("apex_contained", False)
                          and latest["r2y"] >= MIN_R2_Y_INFRAME
                          and latest["ay"]  >  MIN_A_PX_S2)
                # Path C (SUST): pending has been alive long enough AND the
                # latest fit STILL says outbound+fast.  Real throws stay
                # thrown; fakes reverse direction and drop out of "thrown".
                path_c = (pending_age >= SUSTAINED_CONFIRM_S
                          and info is not None
                          and info.get("thrown", False))
                if path_a or path_b or path_c:
                    rep_count += 1
                    cadence.add(now)
                    confirm_info = latest if (path_b or path_c) else pending_info
                    held_info = confirm_info
                    label_hold_until = now + LABEL_HOLD_S
                    if confirm_info.get("burst"):
                        how = "BRST"
                    elif path_a:
                        how = "OOF "
                    elif path_b:
                        how = "APEX"
                    else:
                        how = "SUST"
                    print(f"[{now:7.2f}s] REP #{rep_count} ({how}): "
                          f"vx={confirm_info['vx']:+.0f}px/s "
                          f"vmax={confirm_info['peak_speed']:.0f}px/s "
                          f"r={confirm_info['r_med']:.0f}px "
                          f"R2y={confirm_info['r2y']:.2f} "
                          f"n={confirm_info['n']}")
                    logger.log_throw(now, rep_count, confirm_info,
                                     pose.serialize(pose_info))
                    trail.clear()
                    pending_info = None
                    drop_armed = True
                    cooldown_until = now + REP_COOLDOWN_S
                elif now - pending_set_at >= PENDING_WINDOW_S:
                    pending_info = None     # candidate timed out

            # --- Drop detection (cadence OR position) --------------------
            # CAD: current gap > 2x median of recent inter-rep intervals.
            # POS: ball last seen low in frame (ground zone) AND lost for
            #      at least GROUND_LOST_GRACE_S.  Either signal can fire a
            #      drop; when both agree it's tagged BOTH.
            if drop_armed:
                cad_drop = cadence.gap_is_drop(now)
                pos_drop = None
                if last_seen_pos is not None:
                    cy_last = last_seen_pos[1]
                    in_ground = cy_last >= FRAME_H * GROUND_ZONE_Y_FRAC
                    lost_for  = now - last_seen_at
                    if in_ground and lost_for >= GROUND_LOST_GRACE_S:
                        pos_drop = (lost_for, cy_last)

                if cad_drop is not None or pos_drop is not None:
                    drop_count += 1
                    if cad_drop is not None and pos_drop is not None:
                        tag = "BOTH"
                    elif pos_drop is not None:
                        tag = "POS "
                    else:
                        tag = "CAD "
                    gap = cad_drop[0] if cad_drop else pos_drop[0]
                    logger.log_drop(now, gap)
                    if cad_drop is not None:
                        med, thr = cad_drop[1], cad_drop[2]
                        cad_str = f"cad={med:.2f}s thr={thr:.2f}s"
                    else:
                        cad_str = "cad=n/a"
                    if pos_drop is not None:
                        cy_last = pos_drop[1]
                        pos_str = (f"last cy={cy_last:.0f}/"
                                   f"{FRAME_H} (ground)")
                    else:
                        pos_str = ""
                    print(f"[{now:7.2f}s] DROP #{drop_count} ({tag}) "
                          f"gap={gap:.2f}s  {cad_str}  {pos_str}")
                    drop_armed = False     # rearm on the next rep

            # --- HUD (only on display frames) ----------------------------
            cpm = cadence.rate(now)
            last_ago = cadence.last_throw_ago(now)
            hold_active = held_info is not None and now <= label_hold_until

            if not do_draw:
                continue   # skip drawing + display on this loop iteration

            if hold_active:
                cv2.putText(frame, "REP confirmed", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,  
                            (0, 255, 0), 2)
            elif pending_info is not None:
                cv2.putText(frame, "throwing? waiting for release",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 200, 255), 2)
            elif info is not None:
                cv2.putText(frame,
                            f"tracking  vx={info['vx']:+.0f}",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 2)

            def draw_chip(text, x_right, y_top, fg, scale=0.9):
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
                cv2.rectangle(frame,
                              (x_right - tw - 14, y_top - 4),
                              (x_right,           y_top + th + 8),
                              (0, 0, 0), -1)
                cv2.putText(frame, text,
                            (x_right - tw - 8, y_top + th),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2)

            draw_chip(f"reps: {rep_count}",   FRAME_W - 4,  8, (0, 255, 0),  0.9)
            draw_chip(f"cpm: {cpm:.0f}",      FRAME_W - 4, 48, (0, 220, 255), 0.7)
            draw_chip(f"drops: {drop_count}", FRAME_W - 4, 82, (0,  80, 255), 0.6)

            cv2.putText(frame,
                        f"pts={len(trail)}  wall={args.wall_side}  "
                        f"fps={cur_fps:.1f}",
                        (8, FRAME_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            if last_ago is not None:
                cv2.putText(frame,
                            f"last rep: {last_ago:.1f}s ago",
                            (FRAME_W - 220, FRAME_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

            # Ground-zone line (drops fire if ball is last seen below this).
            gz_y = int(FRAME_H * GROUND_ZONE_Y_FRAC)
            cv2.line(frame, (0, gz_y), (FRAME_W, gz_y), (0, 80, 200), 1)
            cv2.putText(frame, "ground zone", (8, gz_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 200), 1)

            # Small wall-side arrow at the mid-height edge.
            y_arr = FRAME_H // 2
            if wall_dir > 0:
                cv2.arrowedLine(frame, (FRAME_W - 50, y_arr),
                                (FRAME_W - 10, y_arr),
                                (0, 200, 255), 3, tipLength=0.4)
            else:
                cv2.arrowedLine(frame, (50, y_arr), (10, y_arr),
                                (0, 200, 255), 3, tipLength=0.4)

            # HSV calibration crosshair — hold the ball here & press 'h'.
            cx_g, cy_g = FRAME_W // 2, FRAME_H // 2
            b = HSV_SAMPLE_BOX_PX // 2
            cv2.rectangle(frame, (cx_g - b, cy_g - b), (cx_g + b, cy_g + b),
                          (180, 180, 180), 1)
            cv2.putText(frame,
                        "[h] HSV calibrate    [d] reset HSV",
                        (cx_g - 140, cy_g - b - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Transient status banner (HSV updated, AE locked, etc.)
            if hsv_status and time.time() <= hsv_status_until:
                cv2.putText(frame, hsv_status, (8, FRAME_H - 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 220, 255), 2)

            if display is not None:
                display.show("wall ball", frame)
                display.show("mask", mask)
                k = display.get_key()
                if k == ord('q'):
                    break
                if k == ord('r'):
                    rep_count = 0
                    drop_count = 0
                    cadence.reset()
                    trail.clear()
                    pending_info = None
                    held_info = None
                    drop_armed = False
                    cooldown_until = 0.0
                    last_seen_pos = None
                    last_known_r  = None
                    filter_cx.reset()
                    filter_cy.reset()
                    filter_r.reset()
                    print("session reset")
                if k == ord('s'):
                    fn = f"shot_{int(time.time())}.png"
                    cv2.imwrite(fn, frame)
                    print("saved", fn)
                if k == ord('h'):
                    # Sample HSV from the current detection (or center box).
                    sampled = sample_hsv_from(frame, best)
                    if sampled is not None:
                        lo, hi = sampled
                        HSV_LOW_A[:]  = lo
                        HSV_HIGH_A[:] = hi
                        save_hsv_calibration(HSV_LOW_A, HSV_HIGH_A)
                        hsv_status = (f"HSV updated: H={lo[0]}-{hi[0]} "
                                      f"S={lo[1]}-{hi[1]}  V={lo[2]}-{hi[2]}")
                        hsv_status_until = now + 3.0
                        print(f"[{now:7.2f}s] {hsv_status}")
                    else:
                        print("HSV sample: not enough pixels (hold ball "
                              "in center or wait for tracking)")
                if k == ord('d'):
                    HSV_LOW_A[:]  = HSV_LOW_A_DEFAULT
                    HSV_HIGH_A[:] = HSV_HIGH_A_DEFAULT
                    try:
                        os.remove(HSV_CALIB_PATH)
                    except FileNotFoundError:
                        pass
                    hsv_status = "HSV reset to defaults"
                    hsv_status_until = now + 2.0
                    print(f"[{now:7.2f}s] {hsv_status}")
    finally:
        elapsed = time.time() - logger.started_at
        avg_cpm = (rep_count * 60.0 / max(elapsed, 1e-3)) if rep_count else 0.0
        summary = {
            "duration_s":      f"{elapsed:.1f}",
            "reps":            rep_count,
            "drops":           drop_count,
            "avg_cpm":         f"{avg_cpm:.1f}",
            "wall_side":       args.wall_side,
        }
        logger.close(summary)
        try:
            cv_worker.stop()
        except Exception:
            pass
        release()
        if display is not None:
            display.stop()
        else:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print()
        print("=== Session summary ===")
        for k, v in summary.items():
            print(f"  {k:14s} {v}")
        print(f"  log:           {log_path}")


if __name__ == "__main__":
    main()
