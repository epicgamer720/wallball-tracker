"""
Lacrosse ball tracker — light enough for a Raspberry Pi.

Pipeline:
  frame -> blur -> HSV mask -> morph open/close -> most ball-shaped contour
        -> minEnclosingCircle centroid -> append (t, x, y) to trail
        -> fit y(t) = a*t^2 + b*t + c  and  x(t) = m*t + k on a suffix
           window of the trail (longest suffix that fits a parabola)
        -> "thrown" if a > MIN_A_PX_S2 (gravity in image, y grows downward)
           and both fits have good R^2.

Default HSV range targets a yellow ball. Change HSV_LOW/HSV_HIGH below to
retune for a different colour.

Runs from a USB cam, the CSI camera via picamera2, or a video file.

Keys (when a display is attached):
    q  quit
    r  reset trail
    s  save annotated screenshot
"""
import argparse
import collections
import math
import time

import numpy as np
import cv2

# --- Tunables ------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480

# Yellow lacrosse ball — two ranges OR'd:
#   A) the saturated ball body (covers the shadow side via low V floor)
#   B) ONLY the bright specular highlight (very bright + slightly yellow-tinted)
# Skin tones live at H<15 and don't get bright enough to hit B, so they
# don't leak into either range.
HSV_LOW_A  = np.array([15, 100,  60], dtype=np.uint8)
HSV_HIGH_A = np.array([42, 255, 255], dtype=np.uint8)
HSV_LOW_B  = np.array([15,  20, 235], dtype=np.uint8)
HSV_HIGH_B = np.array([42,  95, 255], dtype=np.uint8)

MIN_RADIUS_PX     = 8
MAX_RADIUS_PX     = 220     # close-up balls can fill a large fraction of frame
MIN_CIRCULARITY   = 0.55    # 4*pi*A/P^2; 1.0 is a perfect circle
TRAIL_MAX_AGE_S   = 1.0     # forget points older than this
TRAIL_LEN         = 120
MIN_POINTS_FIT    = 9       # underdetermined small windows give bogus fits
MIN_DURATION_S    = 0.28
MIN_TOTAL_DISP_PX = 120     # at REF_RADIUS_PX; scaled by ball size at fit time
MIN_A_PX_S2       = 400.0   # at REF_RADIUS_PX; scaled by ball size at fit time
MIN_R2_Y          = 0.92
MIN_R2_X          = 0.85
MIN_PEAK_SPEED_PX = 280.0   # at REF_RADIUS_PX; scaled by ball size at fit time
REF_RADIUS_PX     = 30.0    # the ball radius the px thresholds were tuned at
MIN_SCALE         = 0.3     # floor on the distance-scale factor (avoid 0)
LABEL_HOLD_S      = 1.5     # keep "THROWN" label visible this long after detection

# Two paths to confirm a candidate throw:
#   A) ball goes out of view (left frame / motion-blurred / occluded) within
#      PENDING_WINDOW_S of the parabolic fit — caught via RELEASE_GAP_S.
#   B) the visible arc clearly crossed an apex with very tight gravity fit
#      (R^2 >= MIN_R2_Y_INFRAME).  Pump-fakes can't satisfy B because their
#      y(t) isn't a clean quadratic — arm motion has variable acceleration.
PENDING_WINDOW_S  = 0.6     # how long to wait for release after parabolic fit
RELEASE_GAP_S     = 0.18    # tracking gap that confirms ball is gone
MIN_R2_Y_INFRAME  = 0.95    # tighter R^2 required for in-view confirmation
APEX_EDGE_MARGIN_S = 0.05   # apex must be at least this far from window edges
# -------------------------------------------------------------------------


def open_camera(w, h):
    """Try picamera2 first (CSI cam on Pi), fall back to USB/webcam."""
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

    import sys
    backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF),
                ("ANY",   cv2.CAP_ANY)] \
        if sys.platform.startswith("win") else [("ANY", cv2.CAP_ANY)]
    # Camera driver can take several seconds to release after a previous run.
    # Try the whole backend list a few times with a delay between attempts.
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
                time.sleep(0.8)             # warmup; first frame often fails
                ok = False
                for _ in range(30):
                    ok, _frame = cap.read()
                    if ok:
                        break
                    time.sleep(0.1)
                if ok:
                    def make_read(c):
                        def _read():
                            for _ in range(3):
                                ok, frame = c.read()
                                if ok:
                                    return True, frame
                                time.sleep(0.02)
                            return False, None
                        return _read
                    return make_read(cap), cap.release, f"{name}:{idx}"
                last_err = f"{name}:{idx} opened but no frames"
                cap.release()
        if attempt < 3:
            time.sleep(1.5)                 # let the driver fully release
    raise RuntimeError(f"No camera available. {last_err or ''}")


def r2_score(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0


def _fit_window(window):
    """Fit one suffix of the trail. Returns result dict or None.
    Each window entry is (t, x, y, r) with r being the detected ball radius."""
    t = np.array([p[0] for p in window], dtype=np.float64)
    x = np.array([p[1] for p in window], dtype=np.float64)
    y = np.array([p[2] for p in window], dtype=np.float64)
    r = np.array([p[3] for p in window], dtype=np.float64)
    t -= t[0]
    if t[-1] < MIN_DURATION_S:
        return None

    # Distance scale: pixel quantities (displacement, speed, gravity) all
    # scale linearly with ball radius (everything maps through f/Z). Tune the
    # absolute thresholds for REF_RADIUS_PX and scale at fit time.
    r_med = float(np.median(r))
    scale = max(r_med / REF_RADIUS_PX, MIN_SCALE)
    eff_min_disp  = MIN_TOTAL_DISP_PX * scale
    eff_min_a     = MIN_A_PX_S2       * scale
    eff_min_speed = MIN_PEAK_SPEED_PX * scale

    if float(np.hypot(x[-1] - x[0], y[-1] - y[0])) < eff_min_disp:
        return None

    # Peak instantaneous speed — real throws are fast, hand drift isn't.
    dts = np.diff(t)
    dxs = np.diff(x)
    dys = np.diff(y)
    safe_dt = np.where(dts > 1e-6, dts, 1e-6)
    speeds = np.hypot(dxs, dys) / safe_dt
    peak_speed = float(np.max(speeds)) if speeds.size else 0.0
    if peak_speed < eff_min_speed:
        return None

    ay, by, cy = np.polyfit(t, y, 2)
    mx, kx     = np.polyfit(t, x, 1)
    r2y = r2_score(y, np.polyval([ay, by, cy], t))
    # If the throw is near-vertical, x has tiny variance and R^2_x is
    # noise-driven and meaningless. Skip the x-fit gate in that case.
    near_vertical = float(np.std(x)) < 4.0 * scale
    r2x = 1.0 if near_vertical else r2_score(x, np.polyval([mx, kx], t))
    thrown = (ay > eff_min_a) and (r2y > MIN_R2_Y) and (r2x > MIN_R2_X)

    # Apex containment: did the trajectory actually cross the parabola's vertex
    # within the visible window?  Real tosses do; pump-fakes usually do not.
    apex_contained = False
    if ay > 1e-6:
        t_apex = -by / (2.0 * ay)
        apex_contained = (APEX_EDGE_MARGIN_S < t_apex <
                          t[-1] - APEX_EDGE_MARGIN_S)

    return {
        "thrown": bool(thrown),
        "ay": float(ay), "r2y": float(r2y), "r2x": float(r2x),
        "peak_speed": peak_speed,
        "duration": float(t[-1]),
        "fit_y": (float(ay), float(by), float(cy)),
        "fit_x": (float(mx), float(kx)),
        "n": len(window),
        "apex_contained": bool(apex_contained),
        "r_med": r_med,
        "scale": scale,
        "eff_min_a": eff_min_a,
    }


def analyse_trajectory(points):
    """Try suffix windows of the trail, biggest first. Return the largest
    suffix that fits a parabolic throw, or the most recent fit for display."""
    n = len(points)
    if n < MIN_POINTS_FIT:
        return None
    last_valid = None
    for size in range(n, MIN_POINTS_FIT - 1, -1):
        result = _fit_window(points[-size:])
        if result is None:
            continue
        if result["thrown"]:
            return result
        if last_valid is None:
            last_valid = result
    return last_valid


def draw_fitted_curve(frame, t_span, fit_x, fit_y, color=(0, 200, 255)):
    ts = np.linspace(0.0, t_span, 60)
    mx, kx = fit_x
    ay, by, cy = fit_y
    xs = mx * ts + kx
    ys = ay * ts * ts + by * ts + cy
    poly = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(frame, [poly], False, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-display", action="store_true",
                    help="headless: skip cv2.imshow (Pi without monitor)")
    ap.add_argument("--video", type=str, default=None,
                    help="read from a file instead of the camera")
    args = ap.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {args.video}")
        read, release, src = cap.read, cap.release, "file"
    else:
        read, release, src = open_camera(FRAME_W, FRAME_H)
    print(f"source: {src}")

    trail = collections.deque(maxlen=TRAIL_LEN)
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    start = time.time()
    label_hold_until = 0.0
    held_info = None
    throw_count = 0
    pending_info = None      # parabolic fit waiting for release confirmation
    pending_set_at = 0.0
    last_seen_at = 0.0       # last frame the ball was actually detected

    while True:
        ok, frame = read()
        if not ok:
            break
        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        now = time.time() - start

        # mask: ball body (A) OR'd with bright specular highlight (B)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_a = cv2.inRange(hsv, HSV_LOW_A, HSV_HIGH_A)
        mask_b = cv2.inRange(hsv, HSV_LOW_B, HSV_HIGH_B)
        mask = cv2.bitwise_or(mask_a, mask_b)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # forget old points
        while trail and (now - trail[0][0]) > TRAIL_MAX_AGE_S:
            trail.popleft()

        # Pick the most ball-shaped white blob (size + circularity + roundness).
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < math.pi * MIN_RADIUS_PX * MIN_RADIUS_PX * 0.5:
                continue
            perim = cv2.arcLength(c, True)
            if perim < 1.0:
                continue
            circularity = 4.0 * math.pi * area / (perim * perim)
            if circularity < MIN_CIRCULARITY:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if not (MIN_RADIUS_PX <= r <= MAX_RADIUS_PX):
                continue
            score = circularity * math.sqrt(area)
            if score > best_score:
                best_score = score
                best = (float(cx), float(cy), float(r))
        if best is not None:
            cx, cy, r = best
            trail.append((now, cx, cy, r))
            last_seen_at = now
            cv2.circle(frame, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # draw trail
        for i in range(1, len(trail)):
            cv2.line(frame,
                     (int(trail[i-1][1]), int(trail[i-1][2])),
                     (int(trail[i][1]),   int(trail[i][2])),
                     (255, 255, 255), 2)

        info = analyse_trajectory(list(trail))

        # Open a pending candidate the first time the analyser reports a throw.
        if info and info["thrown"] and pending_info is None:
            pending_info  = info
            pending_set_at = now

        # Resolve pending: confirm if the ball disappears (path A) or if the
        # visible arc clearly crossed an apex under gravity (path B).
        # If neither happens within PENDING_WINDOW_S, discard as a fake.
        if pending_info is not None:
            latest = info if info is not None else pending_info
            lost_for = now - last_seen_at
            path_a = lost_for >= RELEASE_GAP_S
            path_b = (latest.get("apex_contained", False)
                      and latest["r2y"] >= MIN_R2_Y_INFRAME
                      and latest["ay"] > MIN_A_PX_S2)
            if path_a or path_b:
                throw_count += 1
                confirm_info = latest if path_b else pending_info
                held_info = confirm_info
                label_hold_until = now + LABEL_HOLD_S
                how = "OOF " if path_a else "APEX"
                print(f"[{now:6.2f}s] THROW #{throw_count} ({how}): "
                      f"ay={confirm_info['ay']:.0f}/"
                      f"{confirm_info['eff_min_a']:.0f}  "
                      f"vmax={confirm_info['peak_speed']:.0f}px/s  "
                      f"R2y={confirm_info['r2y']:.2f}  "
                      f"R2x={confirm_info['r2x']:.2f}  n={confirm_info['n']}  "
                      f"r={confirm_info['r_med']:.0f}px "
                      f"(scale={confirm_info['scale']:.2f})")
                trail.clear()
                pending_info = None
            elif now - pending_set_at >= PENDING_WINDOW_S:
                # ball stayed visible too long without crossing an apex
                # under gravity -> probably a fake.
                pending_info = None

        # Display state, in priority order: thrown > pending > live fit > none.
        hold_active = held_info is not None and now <= label_hold_until
        if hold_active:
            draw_fitted_curve(frame, held_info["duration"],
                              held_info["fit_x"], held_info["fit_y"],
                              color=(0, 255, 0))
            cv2.putText(frame,
                        f"THROWN  ay={held_info['ay']:.0f}px/s^2  "
                        f"R2y={held_info['r2y']:.2f}  R2x={held_info['r2x']:.2f}  "
                        f"n={held_info['n']}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        elif pending_info is not None:
            draw_fitted_curve(frame, pending_info["duration"],
                              pending_info["fit_x"], pending_info["fit_y"],
                              color=(0, 200, 255))
            cv2.putText(frame,
                        f"throwing? waiting for release  "
                        f"ay={pending_info['ay']:.0f}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 200, 255), 2)
        elif info is not None:
            draw_fitted_curve(frame, info["duration"],
                              info["fit_x"], info["fit_y"],
                              color=(180, 180, 180))
            cv2.putText(frame,
                        f"tracking...  ay={info['ay']:.0f}  "
                        f"R2y={info['r2y']:.2f}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 200), 2)
        else:
            cv2.putText(frame, "no fit yet", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)

        cv2.putText(frame, f"pts={len(trail)}", (8, FRAME_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        count_text = f"throws: {throw_count}"
        (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (FRAME_W - tw - 18, 8),
                      (FRAME_W - 4, 14 + th + 6), (0, 0, 0), -1)
        cv2.putText(frame, count_text, (FRAME_W - tw - 12, 14 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not args.no_display:
            cv2.imshow("ball",  frame)
            cv2.imshow("mask",  mask)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('r'):
                trail.clear()
                throw_count = 0
                held_info = None
                pending_info = None
                label_hold_until = 0.0
                print("trail and counter reset")
            if k == ord('s'):
                fn = f"shot_{int(time.time())}.png"
                cv2.imwrite(fn, frame)
                print("saved", fn)

    release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
