"""Parse wall-ball session CSVs into JSON-able analytics.

Pure / no camera or OpenCV deps, so it's testable on any machine and is the
data source for the web dashboard's Analytics tab.

CSV layout written by SessionLogger (wallball.py):
    header:  wall_time_iso,session_t,event,rep_idx,ay_pxps2,vx_pxps,
             peak_speed_pxps,r2y,r2x,n_points,ball_radius_px,pose_blob
    rows:    one per 'throw' and 'drop'  (drop carries gap=Xs in pose_blob)
    footer:  blank line, '# SUMMARY', then '# key,value' rows
"""
import csv
import glob
import math
import os
import re
from datetime import datetime

SESSION_RE = re.compile(r"^wallball_\d{8}_\d{6}$")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def list_session_files(log_dir="sessions"):
    return sorted(glob.glob(os.path.join(log_dir, "wallball_*.csv")))


def session_path(sid, log_dir="sessions"):
    """Resolve a session id to a CSV path, guarding against traversal."""
    if not SESSION_RE.match(sid or ""):
        return None
    p = os.path.join(log_dir, sid + ".csv")
    if not os.path.isfile(p):
        return None
    # ensure it really lives in log_dir
    if os.path.dirname(os.path.abspath(p)) != os.path.abspath(log_dir):
        return None
    return p


def _dt_from_name(path):
    base = os.path.basename(path)
    try:
        return datetime.strptime(base[len("wallball_"):-4], "%Y%m%d_%H%M%S")
    except Exception:
        return None


def _read_rows(path):
    """Return (throws, drops, summary) raw lists/dict."""
    throws, drops, summary = [], [], {}
    header_seen = False
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            c0 = row[0].strip()
            if c0.startswith("#"):
                if c0 == "# SUMMARY":
                    continue
                key = c0.lstrip("#").strip()
                summary[key] = row[1].strip() if len(row) > 1 else ""
                continue
            if not header_seen:
                if row[0] == "wall_time_iso":
                    header_seen = True
                continue
            ev = row[2] if len(row) > 2 else ""
            t = _f(row[1]) if len(row) > 1 else None
            if ev == "throw":
                throws.append({
                    "t": t,
                    "rep": int(row[3]) if len(row) > 3 and row[3] else None,
                    "ay": _f(row[4]) if len(row) > 4 else None,
                    "vx": _f(row[5]) if len(row) > 5 else None,
                    "peak_speed": _f(row[6]) if len(row) > 6 else None,
                    "r2y": _f(row[7]) if len(row) > 7 else None,
                    "r2x": _f(row[8]) if len(row) > 8 else None,
                    "n": int(row[9]) if len(row) > 9 and row[9] else None,
                    "radius": _f(row[10]) if len(row) > 10 else None,
                })
            elif ev == "drop":
                gap = None
                blob = row[11] if len(row) > 11 else ""
                m = re.search(r"gap=([0-9.]+)", blob or "")
                if m:
                    gap = _f(m.group(1))
                drops.append({"t": t, "gap": gap})
    return throws, drops, summary


def _consistency(intervals):
    """Return (mean, std, consistency_pct). Consistency = 100*(1 - CV)."""
    vals = [i for i in intervals if i is not None and i > 0]
    if len(vals) < 2:
        return (vals[0] if vals else None), 0.0 if vals else None, None
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(var)
    cv = std / mean if mean else 0.0
    return mean, std, max(0.0, min(100.0, 100.0 * (1.0 - cv)))


def parse_session(path):
    """Full per-session analytics (events + derived series + stats)."""
    throws, drops, summary = _read_rows(path)
    dt = _dt_from_name(path)

    rep_times = [x["t"] for x in throws if x["t"] is not None]
    intervals = [rep_times[i] - rep_times[i - 1]
                 for i in range(1, len(rep_times))]

    cumulative = [{"t": round(t, 2), "reps": i + 1}
                  for i, t in enumerate(rep_times)]
    cadence_instant = [
        {"t": round(rep_times[i], 2),
         "cpm": round(60.0 / iv, 1) if iv and iv > 0 else 0.0}
        for i, iv in enumerate(intervals, start=1)
    ]
    win = 30.0
    cadence_rolling = []
    for t in rep_times:
        cnt = sum(1 for tt in rep_times if t - win < tt <= t)
        span = min(t, win) if t > 0 else win
        cadence_rolling.append(
            {"t": round(t, 2), "cpm": round(cnt * 60.0 / span, 1) if span else 0.0})

    mean_iv, std_iv, consistency = _consistency(intervals)
    speeds = [x["peak_speed"] for x in throws if x["peak_speed"]]

    duration = _f(summary.get("duration_s"))
    if not duration:
        duration = rep_times[-1] if rep_times else 0.0
    reps = int(summary["reps"]) if str(summary.get("reps", "")).isdigit() else len(throws)
    n_drops = int(summary["drops"]) if str(summary.get("drops", "")).isdigit() else len(drops)
    avg_cpm = _f(summary.get("avg_cpm"))
    if avg_cpm is None:
        avg_cpm = (reps * 60.0 / duration) if duration else 0.0

    return {
        "id": os.path.basename(path)[:-4],
        "datetime": dt.isoformat() if dt else None,
        "date_label": dt.strftime("%b %d, %Y  %H:%M") if dt else os.path.basename(path),
        "summary": {
            "duration_s": round(duration, 1),
            "reps": reps,
            "drops": n_drops,
            "avg_cpm": round(avg_cpm, 1),
            "wall_side": summary.get("wall_side", ""),
        },
        "throws": throws,
        "drops": drops,
        "series": {
            "cumulative_reps": cumulative,
            "cadence_instant": cadence_instant,
            "cadence_rolling": cadence_rolling,
            "peak_speed": [{"rep": x["rep"], "v": round(x["peak_speed"], 0)}
                           for x in throws if x["peak_speed"]],
        },
        "stats": {
            "interval_mean_s": round(mean_iv, 2) if mean_iv else None,
            "interval_std_s": round(std_iv, 2) if std_iv is not None else None,
            "consistency_pct": round(consistency, 0) if consistency is not None else None,
            "peak_speed_avg": round(sum(speeds) / len(speeds), 0) if speeds else None,
        },
    }


def session_brief(path):
    """Light summary for the history list (no big event arrays)."""
    throws, drops, summary = _read_rows(path)
    dt = _dt_from_name(path)
    rep_times = [x["t"] for x in throws if x["t"] is not None]
    intervals = [rep_times[i] - rep_times[i - 1]
                 for i in range(1, len(rep_times))]
    _, _, consistency = _consistency(intervals)
    duration = _f(summary.get("duration_s")) or (rep_times[-1] if rep_times else 0.0)
    reps = int(summary["reps"]) if str(summary.get("reps", "")).isdigit() else len(throws)
    n_drops = int(summary["drops"]) if str(summary.get("drops", "")).isdigit() else len(drops)
    avg_cpm = _f(summary.get("avg_cpm"))
    if avg_cpm is None:
        avg_cpm = (reps * 60.0 / duration) if duration else 0.0
    return {
        "id": os.path.basename(path)[:-4],
        "datetime": dt.isoformat() if dt else None,
        "date_label": dt.strftime("%b %d, %Y  %H:%M") if dt else os.path.basename(path),
        "duration_s": round(duration, 1),
        "reps": reps,
        "drops": n_drops,
        "avg_cpm": round(avg_cpm, 1),
        "consistency_pct": round(consistency, 0) if consistency is not None else None,
    }


def history(log_dir="sessions"):
    """All sessions, newest first, as light summaries."""
    out = []
    for p in list_session_files(log_dir):
        try:
            out.append(session_brief(p))
        except Exception:
            continue
    out.sort(key=lambda s: s["datetime"] or "", reverse=True)
    return out


if __name__ == "__main__":
    import json
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "sessions"
    h = history(d)
    print(f"{len(h)} sessions in {d}/")
    for s in h[:5]:
        print(f"  {s['date_label']}  reps={s['reps']} drops={s['drops']} "
              f"cpm={s['avg_cpm']} consistency={s['consistency_pct']}")
    if h:
        full = parse_session(session_path(h[0]["id"], d))
        print("\nnewest session series keys:", list(full["series"].keys()))
        print("throws:", len(full["throws"]), "drops:", len(full["drops"]))
