"""Web dashboard for the wall-ball tracker.

Three screens served from one Flask app:
  /setup     live preview + mask, HSV sliders / sample-from-ball, exposure + WB
  /live      improved live tracking view (MJPEG) + stat cards + cadence sparkline
  /analytics per-session graphs + cross-session history  (data: analytics.py)

The detection itself is NOT reimplemented here — this module reuses
wallball.py's MaskContourWorker, One-Euro filters, trajectory analysis,
CadenceTracker, SessionLogger and HSV-calibration helpers.  `Engine` just
drives them per-frame and exposes control methods for the web layer.

Run:
    python webui.py                 # http://<host>:8000
    python webui.py --port 8080 --wall-side left
On the Pi it's headless-friendly: no cv2 windows, view from any browser.
"""
import argparse
import collections
import json
import os
import threading
import time
from datetime import datetime

import numpy as np
import cv2
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, url_for)

import wallball as wb
import analytics
import sysinfo
from buttons import ButtonInput

# Resolution / frame-rate presets. Each bundles the resolution-tied tuning so
# a switch stays consistent (ref radius, hard search radius, morph kernels).
RES_PRESETS = {
    "640x480@120": {"label": "640x480 - 120fps", "w": 640,  "h": 480, "fps": 120,
                    "ref": 30.0, "hard": 120.0, "kopen": 3, "kclose": 7},
    "1280x720@60": {"label": "1280x720 - 60fps", "w": 1280, "h": 720, "fps": 60,
                    "ref": 60.0, "hard": 220.0, "kopen": 5, "kclose": 11},
    "1280x800@30": {"label": "1280x800 - 30fps", "w": 1280, "h": 800, "fps": 30,
                    "ref": 60.0, "hard": 220.0, "kopen": 5, "kclose": 11},
}
RUNTIME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "runtime.json")


# --- Detection engine (wraps wallball's pipeline) ------------------------

class Engine:
    """Owns the per-frame detection state machine + the worker thread.

    Detection runs whenever frames are fed in (so /setup can preview the
    mask while you calibrate).  Rep counting + CSV logging only happen
    between start_session() and stop_session().
    """

    def __init__(self, wall_side="right", log_dir="sessions"):
        self.wall_side = wall_side
        self.wall_dir = +1 if wall_side == "right" else -1
        self.log_dir = log_dir
        self.frame_w, self.frame_h = wb.FRAME_W, wb.FRAME_H

        try:
            cv2.setNumThreads(wb.CV_INTERNAL_THREADS)
        except Exception:
            pass

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.cv_worker = wb.MaskContourWorker(
            k_open, k_close, wb.MIN_RADIUS_PX, wb.MAX_RADIUS_PX,
            wb.MIN_CIRCULARITY, wb.MIN_RADIUS_PX_STRICT,
            wb.MIN_CIRCULARITY_STRICT)

        self.cadence = wb.CadenceTracker()
        self.filter_cx = wb.OneEuroFilter(wb.ONEEURO_FC_MIN, wb.ONEEURO_BETA)
        self.filter_cy = wb.OneEuroFilter(wb.ONEEURO_FC_MIN, wb.ONEEURO_BETA)
        self.filter_r = wb.OneEuroFilter(wb.ONEEURO_R_FC_MIN, wb.ONEEURO_R_BETA)
        self.trail = collections.deque(maxlen=wb.TRAIL_LEN)

        self.HSV_LOW_A_DEFAULT = wb.HSV_LOW_A.copy()
        self.HSV_HIGH_A_DEFAULT = wb.HSV_HIGH_A.copy()
        saved = wb.load_hsv_calibration()
        if saved is not None:
            wb.HSV_LOW_A[:], wb.HSV_HIGH_A[:] = saved

        self.lock = threading.Lock()
        self.frame_idx = 0
        self.start = time.time()
        self.fps_window = collections.deque(maxlen=60)
        self.cur_fps = 0.0

        self.logger = None
        self.session_active = False
        self.session_id = None
        self.last_summary = None
        self.status = ""
        self.status_until = 0.0
        self._last_frame = None      # most recent raw frame (for HSV sampling)
        self._last_best = None
        self._reset_state()

    # -- state ------------------------------------------------------------
    def _reset_state(self):
        self.rep_count = 0
        self.drop_count = 0
        self.last_seen_at = 0.0
        self.last_seen_pos = None
        self.last_known_r = None
        self.pending_info = None
        self.pending_set_at = 0.0
        self.held_info = None
        self.label_hold_until = 0.0
        self.drop_armed = False
        self.cooldown_until = 0.0
        self.state_label = "setup"
        self.trail.clear()
        self.cadence.reset()
        self.filter_cx.reset()
        self.filter_cy.reset()
        self.filter_r.reset()

    # -- session control --------------------------------------------------
    def start_session(self):
        with self.lock:
            self.start = time.time()        # rebase session_t to ~0
            self._reset_state()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = f"wallball_{ts}"
            path = os.path.join(self.log_dir, self.session_id + ".csv")
            self.logger = wb.SessionLogger(path)
            self.session_active = True
            self._flash("session started")
            return self.session_id

    def stop_session(self):
        with self.lock:
            if not self.session_active:
                return None
            elapsed = time.time() - self.logger.started_at
            avg_cpm = (self.rep_count * 60.0 / max(elapsed, 1e-3)
                       if self.rep_count else 0.0)
            summary = {
                "duration_s": f"{elapsed:.1f}",
                "reps": self.rep_count,
                "drops": self.drop_count,
                "avg_cpm": f"{avg_cpm:.1f}",
                "wall_side": self.wall_side,
            }
            self.logger.close(summary)
            self.logger = None
            self.session_active = False
            self.last_summary = dict(summary, id=self.session_id)
            self.state_label = "setup"
            self._flash("session saved")
            return self.last_summary

    def reset_counts(self):
        with self.lock:
            keep = self.session_active
            self._reset_state()
            self.session_active = keep
            self._flash("counts reset")

    def _flash(self, msg, secs=3.0):
        self.status = msg
        self.status_until = time.time() + secs

    def set_wall(self, side):
        """Flip which side the wall is on — instant, no camera reopen."""
        self.wall_side = "left" if side == "left" else "right"
        self.wall_dir = +1 if self.wall_side == "right" else -1
        self._flash(f"wall side: {self.wall_side}")

    def reset_tracking(self):
        """Clear position/trail state (e.g. after a resolution change) without
        touching rep/drop counts or an active session."""
        self.trail.clear()
        self.last_seen_pos = None
        self.last_known_r = None
        self.pending_info = None
        self.filter_cx.reset()
        self.filter_cy.reset()
        self.filter_r.reset()
        with self.lock:
            self._last_frame = None
            self._last_best = None

    # -- HSV control ------------------------------------------------------
    def set_hsv(self, low, high):
        wb.HSV_LOW_A[:] = np.array(low, dtype=np.uint8)
        wb.HSV_HIGH_A[:] = np.array(high, dtype=np.uint8)
        wb.save_hsv_calibration(wb.HSV_LOW_A, wb.HSV_HIGH_A)
        self._flash("HSV set")

    def reset_hsv(self):
        wb.HSV_LOW_A[:] = self.HSV_LOW_A_DEFAULT
        wb.HSV_HIGH_A[:] = self.HSV_HIGH_A_DEFAULT
        try:
            os.remove(wb.HSV_CALIB_PATH)
        except FileNotFoundError:
            pass
        self._flash("HSV reset to defaults")

    def sample_hsv(self, source="auto"):
        """source: 'ball' = inside the detected circle (needs a detection),
        'box' = the centre crosshair box, 'auto' = ball if detected else box."""
        with self.lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
            best = self._last_best
        if frame is None:
            return None
        if source == "box":
            best = None
        elif source == "ball" and best is None:
            return None
        sampled = wb.sample_hsv_from(frame, best)
        if sampled is None:
            return None
        lo, hi = sampled
        wb.HSV_LOW_A[:] = lo
        wb.HSV_HIGH_A[:] = hi
        wb.save_hsv_calibration(wb.HSV_LOW_A, wb.HSV_HIGH_A)
        self._flash("HSV sampled from ball")
        return [int(v) for v in lo], [int(v) for v in hi]

    def hsv_bounds(self):
        return {
            "low": [int(v) for v in wb.HSV_LOW_A],
            "high": [int(v) for v in wb.HSV_HIGH_A],
        }

    # -- per-frame --------------------------------------------------------
    def process(self, capture_frame, capture_t, draw=True):
        if (capture_frame.shape[1] != self.frame_w
                or capture_frame.shape[0] != self.frame_h):
            capture_frame = cv2.resize(capture_frame,
                                       (self.frame_w, self.frame_h))

        anchor = ghost = None
        if self.last_seen_pos is not None:
            age = capture_t - self.last_seen_at
            if age < wb.TRACK_LOST_GAP_S and self.last_known_r is not None:
                anchor = (self.last_seen_pos[0], self.last_seen_pos[1],
                          self.last_known_r)
            elif age < wb.GHOST_ANCHOR_GAP_S:
                ghost = (self.last_seen_pos[0], self.last_seen_pos[1])
        self.cv_worker.submit(capture_frame, self.frame_idx, capture_t,
                              anchor, ghost)
        self.frame_idx += 1
        result = self.cv_worker.get_latest()
        if result is None:
            return None
        frame, _wi, now, mask, best = result

        with self.lock:
            self._last_frame = frame
            self._last_best = best

        self.fps_window.append(now)
        if len(self.fps_window) >= 2:
            self.cur_fps = (len(self.fps_window) - 1) / max(
                self.fps_window[-1] - self.fps_window[0], 1e-6)

        while self.trail and (now - self.trail[0][0]) > wb.TRAIL_MAX_AGE_S:
            self.trail.popleft()

        if best is not None:
            cx = self.filter_cx.filter(best[0], now)
            cy = self.filter_cy.filter(best[1], now)
            rr = self.filter_r.filter(best[2], now)
            self.trail.append((now, cx, cy, rr))
            self.last_seen_at = now
            self.last_seen_pos = (cx, cy)
            self.last_known_r = rr
        elif (self.last_seen_at > 0
              and (now - self.last_seen_at) > wb.FILTER_RESET_GAP_S):
            self.filter_cx.reset()
            self.filter_cy.reset()
            self.filter_r.reset()

        if best is not None or self.pending_info is not None:
            info = wb.analyse_trajectory(list(self.trail), self.wall_dir)
        else:
            info = None

        # rep + drop state machine — only while a session is running
        if self.session_active:
            self._run_state_machine(info, now)

        cpm = self.cadence.rate(now)
        last_ago = self.cadence.last_throw_ago(now)
        hold_active = (self.held_info is not None and now <= self.label_hold_until)
        self._update_label(info, hold_active)

        # Detection above runs every frame; the HUD draw (the costly part) is
        # skipped on frames that won't be shown or encoded.
        if draw:
            annotated = self._draw_hud(frame, best, info, now, cpm,
                                       last_ago, hold_active)
        else:
            annotated = frame
        return annotated, mask

    def _run_state_machine(self, info, now):
        if (info and info["thrown"] and self.pending_info is None
                and now >= self.cooldown_until):
            self.pending_info = info
            self.pending_set_at = now

        if self.pending_info is not None:
            latest = info if info is not None else self.pending_info
            lost_for = now - self.last_seen_at
            pending_age = now - self.pending_set_at
            path_a = lost_for >= wb.RELEASE_GAP_S
            path_b = (latest.get("apex_contained", False)
                      and latest["r2y"] >= wb.MIN_R2_Y_INFRAME
                      and latest["ay"] > wb.MIN_A_PX_S2)
            path_c = (pending_age >= wb.SUSTAINED_CONFIRM_S
                      and info is not None and info.get("thrown", False))
            if path_a or path_b or path_c:
                self.rep_count += 1
                self.cadence.add(now)
                confirm = latest if (path_b or path_c) else self.pending_info
                self.held_info = confirm
                self.label_hold_until = now + wb.LABEL_HOLD_S
                if self.logger:
                    self.logger.log_throw(now, self.rep_count, confirm, "")
                self.trail.clear()
                self.pending_info = None
                self.drop_armed = True
                self.cooldown_until = now + wb.REP_COOLDOWN_S
            elif now - self.pending_set_at >= wb.PENDING_WINDOW_S:
                self.pending_info = None

        if self.drop_armed:
            cad_drop = self.cadence.gap_is_drop(now)
            pos_drop = None
            if self.last_seen_pos is not None:
                cy_last = self.last_seen_pos[1]
                in_ground = cy_last >= self.frame_h * wb.GROUND_ZONE_Y_FRAC
                if in_ground and (now - self.last_seen_at) >= wb.GROUND_LOST_GRACE_S:
                    pos_drop = (now - self.last_seen_at, cy_last)
            if cad_drop is not None or pos_drop is not None:
                self.drop_count += 1
                gap = cad_drop[0] if cad_drop else pos_drop[0]
                if self.logger:
                    self.logger.log_drop(now, gap)
                self.drop_armed = False

    def _update_label(self, info, hold_active):
        if hold_active:
            self.state_label = "rep"
        elif self.pending_info is not None:
            self.state_label = "throwing"
        elif info is not None:
            self.state_label = "tracking"
        elif self.session_active:
            self.state_label = "ready"
        else:
            self.state_label = "setup"

    # -- stats for the API ------------------------------------------------
    def get_stats(self):
        with self.lock:
            elapsed = (time.time() - self.start) if self.session_active else 0.0
            return {
                "session_active": self.session_active,
                "session_id": self.session_id,
                "state": self.state_label,
                "reps": self.rep_count,
                "drops": self.drop_count,
                "cpm": round(self.cadence.rate(time.time() - self.start), 1),
                "fps": round(self.cur_fps, 1),
                "elapsed_s": round(elapsed, 1),
                "ball": (None if self.last_seen_pos is None else {
                    "x": round(self.last_seen_pos[0], 1),
                    "y": round(self.last_seen_pos[1], 1),
                    "r": round(self.last_known_r or 0, 1),
                    "fresh": (time.time() - self.start - self.last_seen_at) < 0.4,
                }),
                "hsv": self.hsv_bounds(),
                "status": (self.status if time.time() <= self.status_until else ""),
                "last_summary": self.last_summary,
            }

    # -- HUD --------------------------------------------------------------
    def _chip(self, frame, text, x_right, y_top, fg, scale):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(frame, (x_right - tw - 16, y_top - 6),
                      (x_right, y_top + th + 10), (15, 15, 18), -1)
        cv2.rectangle(frame, (x_right - tw - 16, y_top - 6),
                      (x_right, y_top + th + 10), fg, 1)
        cv2.putText(frame, text, (x_right - tw - 8, y_top + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2, cv2.LINE_AA)

    def _draw_hud(self, frame, best, info, now, cpm, last_ago, hold_active):
        h, w = frame.shape[:2]
        bar = frame[0:44]
        cv2.addWeighted(bar, 0.5, np.full_like(bar, (18, 18, 22)), 0.5, 0, bar)

        if hold_active:
            txt, col = "REP CONFIRMED", (90, 255, 120)
        elif self.pending_info is not None:
            txt, col = "THROWING -- waiting for release", (0, 200, 255)
        elif info is not None:
            txt, col = f"tracking   vx={info['vx']:+.0f}px/s", (210, 210, 210)
        elif self.session_active:
            txt, col = "ready -- throw at the wall", (180, 180, 185)
        else:
            txt, col = "SETUP -- calibrate, then start a session", (180, 180, 185)
        cv2.putText(frame, txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    col, 2, cv2.LINE_AA)

        if len(self.trail) >= 2:
            pts = np.array([(int(p[1]), int(p[2])) for p in self.trail],
                           dtype=np.int32)
            cv2.polylines(frame, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
        if best is not None and self.last_seen_pos is not None:
            cx, cy = int(self.last_seen_pos[0]), int(self.last_seen_pos[1])
            rr = int(self.last_known_r or 0)
            cv2.circle(frame, (cx, cy), max(rr, 3), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)

        self._chip(frame, f"REPS {self.rep_count}", w - 12, 56, (90, 255, 120), 1.0)
        self._chip(frame, f"CPM {cpm:.0f}", w - 12, 98, (0, 220, 255), 0.7)
        self._chip(frame, f"DROPS {self.drop_count}", w - 12, 132, (80, 90, 255), 0.6)

        gz = int(h * wb.GROUND_ZONE_Y_FRAC)
        cv2.line(frame, (0, gz), (w, gz), (60, 70, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, "ground", (8, gz - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (90, 100, 170), 1, cv2.LINE_AA)

        ya = h // 2
        if self.wall_dir > 0:
            cv2.arrowedLine(frame, (w - 55, ya), (w - 15, ya),
                            (0, 200, 255), 3, tipLength=0.4)
        else:
            cv2.arrowedLine(frame, (55, ya), (15, ya),
                            (0, 200, 255), 3, tipLength=0.4)

        if not self.session_active:
            b = wb.HSV_SAMPLE_BOX_PX // 2
            ccx, ccy = w // 2, h // 2
            cv2.rectangle(frame, (ccx - b, ccy - b), (ccx + b, ccy + b),
                          (200, 200, 200), 1)
            cv2.putText(frame, "hold ball here -> Sample", (ccx - 96, ccy - b - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                        cv2.LINE_AA)

        cv2.putText(frame,
                    f"pts={len(self.trail)}  fps={self.cur_fps:.0f}  "
                    f"wall={self.wall_side}",
                    (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (190, 190, 190), 1, cv2.LINE_AA)
        if last_ago is not None:
            cv2.putText(frame, f"last rep {last_ago:.1f}s",
                        (w - 180, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (190, 190, 190), 1, cv2.LINE_AA)
        if self.status and time.time() <= self.status_until:
            cv2.putText(frame, self.status, (12, h - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2,
                        cv2.LINE_AA)
        return frame

    def stop(self):
        try:
            self.cv_worker.stop()
        except Exception:
            pass


# --- On-screen UI (fullscreen on the Pi monitor, 3-button nav) -----------

class ScreenUI:
    """Fullscreen cv2 UI for the Pi's own display, driven by 3 buttons.

    NEXT cycles tabs · OK is the per-tab primary action · BACK is secondary.
    The keyboard mirrors these (n / space|enter / b) so it works without
    hardware; q or Esc quits.
    """
    TABS = ["Live", "Setup", "Debug", "Last"]
    TOP_H, FOOT_H = 46, 40
    HINTS = {
        "Live":  "NEXT: tab   OK: start/stop session   BACK: reset counts",
        "Setup": "NEXT: tab   OK: sample HSV from ball   BACK: reset HSV",
        "Debug": "NEXT: tab   OK: --   BACK: --",
        "Last":  "NEXT: tab   OK: --   BACK: --",
    }
    KEYMAP = {ord('n'): "NEXT", 9: "NEXT", 83: "NEXT",         # n / Tab / right-arrow
              ord(' '): "OK", 13: "OK",                         # space / enter
              ord('b'): "BACK", 8: "BACK", 81: "BACK"}          # b / backspace / left-arrow

    def __init__(self, runner):
        self.runner = runner
        self.engine = runner.engine
        self.tab = 0
        self.win = "wall ball"
        self._inited = False

    def _init_window(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        self._inited = True

    def handle(self, ev):
        if ev == "NEXT":
            self.tab = (self.tab + 1) % len(self.TABS)
        elif ev == "OK":
            t = self.TABS[self.tab]
            if t == "Live":
                if self.engine.session_active:
                    self.engine.stop_session()
                    self.tab = self.TABS.index("Last")
                else:
                    self.engine.start_session()
            elif t == "Setup":
                self.engine.sample_hsv()
        elif ev == "BACK":
            t = self.TABS[self.tab]
            if t == "Live":
                self.engine.reset_counts()
            elif t == "Setup":
                self.engine.reset_hsv()

    # -- drawing helpers --------------------------------------------------
    @staticmethod
    def _place(canvas, img, y0, y1):
        if img is None:
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bw, bh = canvas.shape[1], y1 - y0
        ih, iw = img.shape[:2]
        s = min(bw / iw, bh / ih)
        nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
        r = cv2.resize(img, (nw, nh))
        ox, oy = (bw - nw) // 2, y0 + (bh - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = r

    @staticmethod
    def _bar(canvas, x, y, w, h, pct, color):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 58, 70), 1)
        fw = int(w * max(0.0, min(100.0, pct)) / 100.0)
        if fw > 0:
            cv2.rectangle(canvas, (x, y), (x + fw, y + h), color, -1)

    @staticmethod
    def _txt(canvas, s, x, y, scale=0.6, color=(225, 225, 230), thick=1):
        cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thick, cv2.LINE_AA)

    def _chrome(self, canvas):
        w = canvas.shape[1]
        h = canvas.shape[0]
        tw = w // len(self.TABS)
        cv2.rectangle(canvas, (0, 0), (w, self.TOP_H), (20, 20, 26), -1)
        for i, name in enumerate(self.TABS):
            x0 = i * tw
            if i == self.tab:
                cv2.rectangle(canvas, (x0, 0), (x0 + tw, self.TOP_H),
                              (99, 102, 241), -1)
            col = (255, 255, 255) if i == self.tab else (160, 162, 175)
            (sw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            self._txt(canvas, name, x0 + (tw - sw) // 2, 31, 0.7, col, 2)
        cv2.rectangle(canvas, (0, h - self.FOOT_H), (w, h), (20, 20, 26), -1)
        self._txt(canvas, self.HINTS[self.TABS[self.tab]], 14, h - 14,
                  0.5, (150, 152, 165), 1)

    # -- per-tab bodies ---------------------------------------------------
    def _live(self, canvas, annotated):
        self._place(canvas, annotated, self.TOP_H, canvas.shape[0] - self.FOOT_H)

    def _setup(self, canvas, mask):
        h = canvas.shape[0]
        self._place(canvas, mask, self.TOP_H, h - self.FOOT_H - 70)
        b = self.engine.hsv_bounds()
        y = h - self.FOOT_H - 44
        self._txt(canvas, f"HSV  H {b['low'][0]}-{b['high'][0]}   "
                  f"S {b['low'][1]}-{b['high'][1]}   V {b['low'][2]}-{b['high'][2]}",
                  14, y, 0.6, (0, 220, 255), 2)
        ball = self.engine.last_seen_pos
        det = (f"ball r={self.engine.last_known_r:.0f}px" if ball is not None
               and self.engine.last_known_r else "no ball — hold it in view")
        self._txt(canvas, det, 14, y + 26, 0.55, (200, 200, 205), 1)

    def _debug(self, canvas):
        d = self.runner.debug_cache or {}
        self._txt(canvas, "SYSTEM", 18, self.TOP_H + 34, 0.8, (235, 235, 240), 2)
        big = f"CPU {d.get('cpu_total', 0):.0f}%   {d.get('cpu_temp', 0):.0f}\xb0C"
        self._txt(canvas, big, 18, self.TOP_H + 74, 1.0, (120, 230, 140), 2)
        cores = d.get("cpu_cores", [])
        y = self.TOP_H + 110
        for i, c in enumerate(cores):
            col = (90, 200, 250) if c < 70 else (90, 120, 250)
            self._txt(canvas, f"C{i}", 18, y + 15, 0.5, (170, 170, 180))
            self._bar(canvas, 56, y, 360, 16, c, col)
            self._txt(canvas, f"{c:.0f}%", 424, y + 14, 0.5, (200, 200, 210))
            y += 24
        mem = d.get("mem", {})
        self._txt(canvas, f"RAM {mem.get('used_mb', 0)}/{mem.get('total_mb', 0)} MB",
                  520, self.TOP_H + 110, 0.6, (225, 225, 230))
        self._bar(canvas, 520, self.TOP_H + 122, 300, 16,
                  mem.get("percent", 0), (250, 180, 90))
        load = d.get("load", [0, 0, 0])
        rows = [
            f"load  {load[0]:.2f}  {load[1]:.2f}  {load[2]:.2f}",
            f"uptime  {d.get('uptime', '?')}",
            f"loop fps  {self.engine.cur_fps:.0f}",
            f"camera  {self.runner.src}  ({'ok' if self.runner.camera_ok else 'no cam'})",
            f"buttons  {self.runner.buttons.status()['detail'] if self.runner.buttons else 'keyboard only'}",
        ]
        yy = self.TOP_H + 170
        for r in rows:
            self._txt(canvas, r, 520, yy, 0.6, (210, 210, 218))
            yy += 30
        temps = d.get("temps", {})
        yy = self.TOP_H + 110 + len(cores) * 24 + 16
        self._txt(canvas, "temps: " + "  ".join(
            f"{k.replace('-thermal', '')} {v:.0f}\xb0" for k, v in temps.items()),
            18, yy, 0.5, (180, 180, 190))

    def _last(self, canvas):
        s = self.engine.last_summary
        self._txt(canvas, "LAST SESSION", 18, self.TOP_H + 40, 0.8,
                  (235, 235, 240), 2)
        if not s:
            self._txt(canvas, "No session saved yet.", 18, self.TOP_H + 90,
                      0.7, (170, 170, 180))
            self._txt(canvas, "On Live, press OK to start one.", 18,
                      self.TOP_H + 124, 0.55, (150, 150, 160))
            return
        rows = [("reps", s.get("reps")), ("drops", s.get("drops")),
                ("duration", f"{s.get('duration_s')}s"),
                ("avg cadence", f"{s.get('avg_cpm')} /min"),
                ("wall", s.get("wall_side"))]
        y = self.TOP_H + 92
        for k, v in rows:
            self._txt(canvas, k, 18, y, 0.7, (160, 162, 175))
            self._txt(canvas, str(v), 300, y, 0.8, (120, 230, 140), 2)
            y += 46

    def render(self, annotated, mask):
        if not self._inited:
            self._init_window()
        canvas = np.full((wb.FRAME_H, wb.FRAME_W, 3), 14, np.uint8)
        t = self.TABS[self.tab]
        if t == "Live":
            self._live(canvas, annotated)
        elif t == "Setup":
            self._setup(canvas, mask)
        elif t == "Debug":
            self._debug(canvas)
        else:
            self._last(canvas)
        self._chrome(canvas)
        cv2.imshow(self.win, canvas)
        return cv2.waitKey(1) & 0xFF


# --- Camera runner (owns capture + applies control commands) -------------

class Runner:
    """Background thread: read camera -> engine.process -> publish JPEGs.

    Reads the camera directly (no ThreadedCapture) so camera-control
    cap.set() calls can be applied safely in the same thread, between reads.
    """

    def __init__(self, wall_side="right", log_dir="sessions", stream_w=960,
                 screen=False, btn_pins=None, screen_fps=20, stream_fps=12):
        self.engine = Engine(wall_side, log_dir)
        self.log_dir = log_dir
        self.stream_w = stream_w
        self.cap = None
        self.raw_read = None
        self.raw_release = None
        self.src = "none"
        self.camera_ok = False
        self.camera_err = ""
        self.is_linux = (os.name != "nt")
        self.running = False
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_mask_jpeg = None
        self._cmd_lock = threading.Lock()
        self._cmds = []
        self._placeholder = self._make_placeholder()
        # debug metrics (dependency-free), refreshed once per second
        self.sysinfo = sysinfo.SysInfo()
        self.debug_cache = {}
        self._last_dbg = 0.0
        # throttle the costly display/encode work; detection still runs every frame
        self._last_encode = 0.0
        self._last_screen = 0.0
        self._encode_dt = 1.0 / max(stream_fps, 1)
        self._screen_dt = 1.0 / max(screen_fps, 1)
        # cache the most recent HUD-drawn frame so the display/stream reuse it
        # when the worker has no fresh result (instead of flashing a placeholder)
        self._last_annotated = None
        self._last_mask = None
        # resolution preset + wall side, persisted across restarts
        self.wall_side = wall_side
        self.current_preset = self._detect_preset()
        self._pending_preset = None
        self._load_runtime()
        # optional on-screen UI on the Pi's monitor + 3-button input
        self.buttons = ButtonInput(btn_pins) if (screen and btn_pins) else None
        self.screen = ScreenUI(self) if screen else None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def _make_placeholder(self, msg="connecting to camera..."):
        img = np.full((wb.FRAME_H, wb.FRAME_W, 3), 24, np.uint8)
        cv2.putText(img, msg, (40, wb.FRAME_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 130), 2,
                    cv2.LINE_AA)
        return img

    # -- resolution presets + wall side ----------------------------------
    def _detect_preset(self):
        for pid, p in RES_PRESETS.items():
            if (p["w"], p["h"]) == (wb.FRAME_W, wb.FRAME_H) and p["fps"] == wb.TARGET_FPS:
                return pid
        return next(iter(RES_PRESETS))

    def _apply_preset_values(self, pid):
        """Set the resolution-tied params (no camera reopen)."""
        p = RES_PRESETS.get(pid)
        if not p:
            return
        wb.FRAME_W, wb.FRAME_H, wb.TARGET_FPS = p["w"], p["h"], p["fps"]
        wb.REF_RADIUS_PX, wb.TRACK_HARD_PX = p["ref"], p["hard"]
        self.engine.frame_w, self.engine.frame_h = p["w"], p["h"]
        self.engine.cv_worker.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (p["kopen"], p["kopen"]))
        self.engine.cv_worker.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (p["kclose"], p["kclose"]))
        self.current_preset = pid

    def _load_runtime(self):
        try:
            with open(RUNTIME_PATH) as f:
                d = json.load(f)
        except Exception:
            return
        if d.get("preset") in RES_PRESETS:
            self._apply_preset_values(d["preset"])
        if d.get("wall") in ("left", "right"):
            self.wall_side = d["wall"]
            self.engine.set_wall(d["wall"])

    def _save_runtime(self):
        try:
            with open(RUNTIME_PATH, "w") as f:
                json.dump({"preset": self.current_preset, "wall": self.wall_side}, f)
        except Exception:
            pass

    def request_resolution(self, pid):
        if pid not in RES_PRESETS:
            return False
        self._pending_preset = pid       # applied in the loop thread
        self.current_preset = pid
        self._save_runtime()
        return True

    def set_wall(self, side):
        self.engine.set_wall(side)
        self.wall_side = self.engine.wall_side
        self._save_runtime()

    def _do_apply_preset(self, pid):
        """Run in the loop thread: reopen the camera at the new resolution."""
        try:
            self.raw_release()
        except Exception:
            pass
        self.camera_ok = False
        time.sleep(1.0)        # let V4L2 fully release before reopening (avoids wedging)
        self._apply_preset_values(pid)
        self.engine.reset_tracking()
        self.engine.start = time.time()
        self._last_annotated = None
        self._last_mask = None
        self._open()
        print(f"[preset] {pid} -> {self.src}", flush=True)

    def _open(self):
        try:
            r, rel, src, cap = wb.open_camera(
                self.engine.frame_w, self.engine.frame_h, return_cap=True)
            self.raw_read, self.raw_release, self.src, self.cap = r, rel, src, cap
            self.camera_ok = True
            self.is_linux = src.startswith("V4L2") or (os.name != "nt")
        except Exception as e:
            self.camera_ok = False
            self.camera_err = str(e)

    def _loop(self):
        self._open()
        last_open_try = time.time()
        while self.running:
            if time.time() - self._last_dbg >= 1.0:
                try:
                    self.debug_cache = self.sysinfo.snapshot()
                except Exception:
                    pass
                self._last_dbg = time.time()

            if self._pending_preset:
                pid = self._pending_preset
                self._pending_preset = None
                self._do_apply_preset(pid)

            frame = None
            if self.camera_ok:
                self._drain_cmds()
                ok, frame = self.raw_read()
                if not ok:
                    frame = None

            now_t = time.time()
            do_encode = (now_t - self._last_encode) >= self._encode_dt
            do_screen = (self.screen is not None
                         and (now_t - self._last_screen) >= self._screen_dt)

            drew = do_encode or do_screen
            if frame is not None:
                out = self.engine.process(frame, now_t - self.engine.start, draw=drew)
                # Only cache HUD-drawn frames, so reuse never shows a frame
                # missing the overlay.
                if out is not None and drew:
                    self._last_annotated, self._last_mask = out
            elif not self.camera_ok:
                self._last_annotated = self._make_placeholder(
                    "no camera: " + self.camera_err[:40])
                self._last_mask = None
                if now_t - last_open_try > 2.0:
                    self._open()
                    last_open_try = now_t

            if do_encode and self._last_annotated is not None:
                self._last_encode = now_t
                ej = self._encode(self._last_annotated)
                em = self._encode(self._last_mask) if self._last_mask is not None else None
                with self.lock:
                    self.latest_jpeg = ej
                    self.latest_mask_jpeg = em

            if do_screen:
                self._last_screen = now_t
                try:
                    if self.buttons is not None:
                        ev = self.buttons.poll()
                        while ev is not None:
                            self.screen.handle(ev)
                            ev = self.buttons.poll()
                    show = (self._last_annotated if self._last_annotated is not None
                            else self._make_placeholder("starting..."))
                    k = self.screen.render(show, self._last_mask)
                    if k in (ord('q'), 27):
                        self.running = False
                        break
                    if k in ScreenUI.KEYMAP:
                        self.screen.handle(ScreenUI.KEYMAP[k])
                except Exception as e:
                    print(f"[screen] disabled ({e}); web UI keeps running",
                          flush=True)
                    self.screen = None

            if not self.camera_ok:
                time.sleep(0.05)     # idle backoff when there's no live frame

    def _encode(self, img):
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.stream_w and img.shape[1] > self.stream_w:
            scale = self.stream_w / img.shape[1]
            img = cv2.resize(img, (self.stream_w, int(img.shape[0] * scale)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    # -- camera control (queued; applied in the capture thread) -----------
    def queue_cmd(self, fn):
        with self._cmd_lock:
            self._cmds.append(fn)

    def _drain_cmds(self):
        with self._cmd_lock:
            cmds, self._cmds = self._cmds, []
        for fn in cmds:
            try:
                fn(self.cap)
            except Exception:
                pass

    def set_auto_exposure(self, on):
        auto_val = 3.0 if self.is_linux else 0.75
        man_val = 1.0 if self.is_linux else 0.25
        self.queue_cmd(lambda c: c.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                                       auto_val if on else man_val))

    def set_exposure(self, val):
        self.queue_cmd(lambda c: c.set(cv2.CAP_PROP_EXPOSURE, float(val)))

    def set_auto_wb(self, on):
        self.queue_cmd(lambda c: c.set(cv2.CAP_PROP_AUTO_WB, 1.0 if on else 0.0))

    def set_wb(self, val):
        self.queue_cmd(lambda c: c.set(cv2.CAP_PROP_WB_TEMPERATURE, float(val)))

    def camera_state(self):
        if not self.camera_ok or self.cap is None:
            return {"ok": False, "src": self.src, "err": self.camera_err,
                    "is_linux": self.is_linux}
        g = self.cap.get
        return {
            "ok": True, "src": self.src, "is_linux": self.is_linux,
            "auto_exposure": g(cv2.CAP_PROP_AUTO_EXPOSURE),
            "exposure": g(cv2.CAP_PROP_EXPOSURE),
            "auto_wb": g(cv2.CAP_PROP_AUTO_WB),
            "wb": g(cv2.CAP_PROP_WB_TEMPERATURE),
        }


# --- Flask app -----------------------------------------------------------

app = Flask(__name__)
# Dev dashboard: always reload edited templates, and never let the browser
# serve stale HTML/CSS/JS (otherwise UI tweaks "look like nothing changed").
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


runner = None      # set in main()


def _mjpeg(kind):
    boundary = b"--frame\r\n"
    while True:
        with runner.lock:
            buf = runner.latest_mask_jpeg if kind == "mask" else runner.latest_jpeg
        if buf is None:
            time.sleep(0.05)
            continue
        yield (boundary + b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
        time.sleep(1 / 30.0)


@app.route("/")
def index():
    return redirect(url_for("setup"))


@app.route("/setup")
def setup():
    return render_template("setup.html", active="setup")


@app.route("/live")
def live():
    return render_template("live.html", active="live")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html", active="analytics")


@app.route("/debug")
def debug_page():
    return render_template("debug.html", active="debug")


@app.route("/stream/video")
def stream_video():
    return Response(_mjpeg("video"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stream/mask")
def stream_mask():
    return Response(_mjpeg("mask"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    st = runner.engine.get_stats()
    st["camera"] = runner.camera_state()
    return jsonify(st)


@app.route("/api/debug")
def api_debug():
    d = dict(runner.debug_cache)
    d["fps"] = round(runner.engine.cur_fps, 1)
    d["camera"] = runner.camera_state()
    d["buttons"] = (runner.buttons.status() if runner.buttons
                    else {"gpio": False, "detail": "keyboard only", "lines": 0})
    d["session_active"] = runner.engine.session_active
    d["preset"] = runner.current_preset
    d["wall"] = runner.wall_side
    return jsonify(d)


@app.route("/api/config")
def api_config():
    return jsonify({
        "preset": runner.current_preset,
        "wall": runner.wall_side,
        "presets": {k: v["label"] for k, v in RES_PRESETS.items()},
    })


@app.route("/api/resolution", methods=["POST"])
def api_resolution():
    pid = (request.get_json(silent=True) or {}).get("preset")
    ok = runner.request_resolution(pid)
    return jsonify({"ok": ok, "preset": runner.current_preset})


@app.route("/api/wall", methods=["POST"])
def api_wall():
    side = (request.get_json(silent=True) or {}).get("side")
    runner.set_wall(side)
    return jsonify({"ok": True, "wall": runner.wall_side})


@app.route("/api/hsv", methods=["GET", "POST"])
def api_hsv():
    if request.method == "POST":
        d = request.get_json(force=True)
        runner.engine.set_hsv(d["low"], d["high"])
    return jsonify(runner.engine.hsv_bounds())


@app.route("/api/hsv/sample", methods=["POST"])
def api_hsv_sample():
    src = (request.get_json(silent=True) or {}).get("source", "auto")
    res = runner.engine.sample_hsv(src)
    if res is None:
        msg = ("no ball detected — point the camera at it first"
               if src == "ball" else
               "not enough pixels — hold the ball in the centre box")
        return jsonify({"ok": False, "msg": msg}), 200
    return jsonify({"ok": True, "low": res[0], "high": res[1]})


@app.route("/api/hsv/reset", methods=["POST"])
def api_hsv_reset():
    runner.engine.reset_hsv()
    return jsonify(runner.engine.hsv_bounds())


@app.route("/api/camera", methods=["POST"])
def api_camera():
    d = request.get_json(force=True)
    if "auto_exposure" in d:
        runner.set_auto_exposure(bool(d["auto_exposure"]))
    if "exposure" in d:
        runner.set_exposure(d["exposure"])
    if "auto_wb" in d:
        runner.set_auto_wb(bool(d["auto_wb"]))
    if "wb" in d:
        runner.set_wb(d["wb"])
    return jsonify({"ok": True})


@app.route("/api/session/start", methods=["POST"])
def api_session_start():
    return jsonify({"ok": True, "id": runner.engine.start_session()})


@app.route("/api/session/stop", methods=["POST"])
def api_session_stop():
    return jsonify({"ok": True, "summary": runner.engine.stop_session()})


@app.route("/api/session/reset", methods=["POST"])
def api_session_reset():
    runner.engine.reset_counts()
    return jsonify({"ok": True})


@app.route("/api/sessions")
def api_sessions():
    return jsonify(analytics.history(runner.log_dir))


@app.route("/api/session/<sid>")
def api_session(sid):
    path = analytics.session_path(sid, runner.log_dir)
    if path is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(analytics.parse_session(path))


def main():
    global runner
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--wall-side", choices=["right", "left"], default="right")
    ap.add_argument("--log-dir", default="sessions")
    ap.add_argument("--screen", action="store_true",
                    help="also draw a fullscreen UI on the Pi's own monitor")
    ap.add_argument("--btn-pins", default=None,
                    help="GPIO buttons 'chip:line,chip:line,chip:line' for "
                         "NEXT,OK,BACK (e.g. 4:18,4:19,4:20); omit for keyboard")
    ap.add_argument("--screen-fps", type=int, default=20,
                    help="on-screen UI redraw rate (lower frees CPU for detection)")
    ap.add_argument("--stream-fps", type=int, default=12,
                    help="web preview JPEG encode rate")
    args = ap.parse_args()

    pins = [p.strip() for p in args.btn_pins.split(",")] if args.btn_pins else None
    runner = Runner(args.wall_side, args.log_dir,
                    screen=args.screen, btn_pins=pins,
                    screen_fps=args.screen_fps, stream_fps=args.stream_fps)
    runner.start()
    print(f"wall-ball web UI -> http://{args.host}:{args.port}  "
          f"(wall={args.wall_side}, screen={args.screen})")
    if runner.buttons:
        print(f"buttons: {runner.buttons.status()}")
    app.run(host=args.host, port=args.port, threaded=True,
            debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
