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
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import cv2


# --- Tunables ------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480

# Yellow ball — two-range HSV (saturated body OR bright specular highlight).
HSV_LOW_A  = np.array([15, 100,  60], dtype=np.uint8)
HSV_HIGH_A = np.array([42, 255, 255], dtype=np.uint8)
HSV_LOW_B  = np.array([15,  20, 235], dtype=np.uint8)
HSV_HIGH_B = np.array([42,  95, 255], dtype=np.uint8)

MIN_RADIUS_PX     = 8
MAX_RADIUS_PX     = 220
MIN_CIRCULARITY   = 0.55       # back where it was — looser broke x-fits
TRAIL_MAX_AGE_S   = 1.0
TRAIL_LEN         = 240        # generous for high-fps cameras

# Parabolic-fit thresholds — loosened so fast, blurry, near-straight wall
# ball throws still produce a candidate.  The speed gates below do the real
# filtering of fake/idle motion.
MIN_POINTS_FIT    = 7
MIN_DURATION_S    = 0.10
MIN_TOTAL_DISP_PX = 120
MIN_A_PX_S2       = 250.0
MIN_R2_Y          = 0.70       # was 0.85
MIN_R2_X          = 0.55       # was 0.80
REF_RADIUS_PX     = 30.0       # px thresholds were tuned at this ball size
MIN_SCALE         = 0.3

# Speed gates — *tightened* to cut slow-arm false positives. Real throws
# blow past these; idle gestures don't.
MIN_PEAK_SPEED_PX  = 500.0      # was 300
MIN_OUTBOUND_VX_PX = 250.0      # was 100

# Burst detector — for lacrosse-stick throws, where the ball is only briefly
# visible mid-flight (it lives in the stick's netting otherwise). Fires
# without a parabolic fit when there's a short burst of fast outbound motion.
MIN_BURST_POINTS        = 3      # was 4 — catch very brief lacrosse releases
MIN_BURST_DURATION_S    = 0.03
MIN_BURST_DX_PX         = 80     # at REF_RADIUS_PX; scaled by ball size
MIN_BURST_SPEED_PX      = 700.0  # at REF_RADIUS_PX; scaled by ball size
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
# -------------------------------------------------------------------------


# --- Camera (same multi-backend logic as tracker.py) ---------------------

def open_camera(w, h):
    try:
        from picamera2 import Picamera2  # type: ignore
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"}))
        cam.start()
        time.sleep(0.2)
        def read():
            arr = cam.capture_array()
            return True, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return read, cam.stop, "picamera2"
    except Exception:
        pass

    backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF),
                ("ANY",   cv2.CAP_ANY)] \
        if sys.platform.startswith("win") else [("ANY", cv2.CAP_ANY)]
    last_err = None
    for attempt in range(4):
        for name, backend in backends:
            for idx in range(3):
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                time.sleep(0.8)
                ok = False
                for _ in range(30):
                    ok, _f = cap.read()
                    if ok:
                        break
                    time.sleep(0.1)
                if ok:
                    def make_read(c):
                        def _read():
                            for _ in range(3):
                                ok, fr = c.read()
                                if ok:
                                    return True, fr
                                time.sleep(0.02)
                            return False, None
                        return _read
                    return make_read(cap), cap.release, f"{name}:{idx}"
                last_err = f"{name}:{idx} opened but no frames"
                cap.release()
        if attempt < 3:
            time.sleep(1.5)
    raise RuntimeError(f"No camera available. {last_err or ''}")


# --- Trajectory analysis -------------------------------------------------

def r2_score(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0


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
    eff_v     = MIN_PEAK_SPEED_PX     * scale
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
    if peak_speed < MIN_BURST_SPEED_PX * scale:
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

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {args.video}")
        read, release, src = cap.read, cap.release, "file"
    else:
        read, release, src = open_camera(FRAME_W, FRAME_H)
    print(f"source: {src}  wall: {args.wall_side}")

    log_path = os.path.join(
        args.log_dir,
        f"wallball_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    logger = SessionLogger(log_path)
    print(f"logging to {log_path}")

    cadence = CadenceTracker()
    pose = PoseEstimator()           # <-- swap in a real estimator here later

    trail = collections.deque(maxlen=TRAIL_LEN)
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    start = time.time()
    rep_count        = 0
    drop_count       = 0
    last_seen_at     = 0.0
    last_seen_pos    = None   # (cx, cy) of most recent ball detection
    pending_info     = None
    pending_set_at   = 0.0
    held_info        = None
    label_hold_until = 0.0
    drop_armed       = False   # set True after each rep; clears after a drop fires
    cooldown_until   = 0.0     # suppress new reps until this session time

    try:
        while True:
            ok, frame = read()
            if not ok:
                break
            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            now = time.time() - start

            # --- Pose hook (no-op by default) ----------------------------
            pose_info = pose(frame) if pose.enabled else None
            if pose_info is not None:
                pose.annotate(frame, pose_info)

            # --- Ball mask -----------------------------------------------
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask_a = cv2.inRange(hsv, HSV_LOW_A, HSV_HIGH_A)
            mask_b = cv2.inRange(hsv, HSV_LOW_B, HSV_HIGH_B)
            mask = cv2.bitwise_or(mask_a, mask_b)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

            # Forget stale trail points.
            while trail and (now - trail[0][0]) > TRAIL_MAX_AGE_S:
                trail.popleft()

            # --- Best ball-shaped contour --------------------------------
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            best, best_score = None, 0.0
            for c in contours:
                area = cv2.contourArea(c)
                if area < math.pi * MIN_RADIUS_PX * MIN_RADIUS_PX * 0.5:
                    continue
                perim = cv2.arcLength(c, True)
                if perim < 1.0:
                    continue
                circ = 4.0 * math.pi * area / (perim * perim)
                if circ < MIN_CIRCULARITY:
                    continue
                (cx, cy), rr = cv2.minEnclosingCircle(c)
                if not (MIN_RADIUS_PX <= rr <= MAX_RADIUS_PX):
                    continue
                score = circ * math.sqrt(area)
                if score > best_score:
                    best_score = score
                    best = (float(cx), float(cy), float(rr))

            if best is not None:
                cx, cy, rr = best
                trail.append((now, cx, cy, rr))
                last_seen_at = now
                last_seen_pos = (cx, cy)
                cv2.circle(frame, (int(cx), int(cy)), int(rr),
                           (0, 255, 255), 2)
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

            # --- Trail polyline ------------------------------------------
            for i in range(1, len(trail)):
                cv2.line(frame,
                         (int(trail[i-1][1]), int(trail[i-1][2])),
                         (int(trail[i][1]),   int(trail[i][2])),
                         (255, 255, 255), 2)

            # --- Throw analysis + release confirmation --------------------
            info = analyse_trajectory(list(trail), wall_dir)

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

            # --- HUD -----------------------------------------------------
            cpm = cadence.rate(now)
            last_ago = cadence.last_throw_ago(now)
            hold_active = held_info is not None and now <= label_hold_until

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
                        f"pts={len(trail)}  wall={args.wall_side}",
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

            if not args.no_display:
                cv2.imshow("wall ball", frame)
                cv2.imshow("mask", mask)
                k = cv2.waitKey(1) & 0xFF
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
                    print("session reset")
                if k == ord('s'):
                    fn = f"shot_{int(time.time())}.png"
                    cv2.imwrite(fn, frame)
                    print("saved", fn)
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
        release()
        cv2.destroyAllWindows()
        print()
        print("=== Session summary ===")
        for k, v in summary.items():
            print(f"  {k:14s} {v}")
        print(f"  log:           {log_path}")


if __name__ == "__main__":
    main()
