"""Dependency-free system metrics for the Debug views (no psutil needed).

Reads /proc and /sys directly, so it works on the Orange Pi out of the box.
CPU% is computed as a delta between successive snapshot() calls, so call it
on a regular cadence (the web Debug tab polls ~1s; the Runner caches it).
"""
import os
import time


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except Exception:
        return None


class SysInfo:
    def __init__(self):
        self._prev = self._cpu_times()
        self._prev_t = time.time()

    # --- CPU -------------------------------------------------------------
    def _cpu_times(self):
        """name -> (idle, total) from /proc/stat (aggregate + per-core)."""
        out = {}
        data = _read("/proc/stat") or ""
        for line in data.splitlines():
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            name = parts[0]
            nums = [int(x) for x in parts[1:]]
            if len(nums) < 5:
                continue
            idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
            total = sum(nums)
            out[name] = (idle, total)
        return out

    def cpu_percent(self):
        cur = self._cpu_times()
        prev = self._prev
        self._prev = cur

        def pct(name):
            if name not in prev or name not in cur:
                return 0.0
            di = cur[name][0] - prev[name][0]
            dt = cur[name][1] - prev[name][1]
            if dt <= 0:
                return 0.0
            return max(0.0, min(100.0, 100.0 * (1.0 - di / dt)))

        cores = sorted([n for n in cur if n != "cpu"],
                       key=lambda n: int(n[3:]))
        return {"total": round(pct("cpu"), 1),
                "cores": [round(pct(n), 1) for n in cores]}

    # --- temps -----------------------------------------------------------
    def temps(self):
        out = {}
        base = "/sys/class/thermal"
        try:
            zones = [z for z in os.listdir(base) if z.startswith("thermal_zone")]
        except Exception:
            return out
        for z in zones:
            typ = (_read(f"{base}/{z}/type") or z).strip()
            raw = _read(f"{base}/{z}/temp")
            if raw is None:
                continue
            try:
                out[typ] = round(int(raw.strip()) / 1000.0, 1)
            except ValueError:
                pass
        return out

    # --- memory ----------------------------------------------------------
    def mem(self):
        data = _read("/proc/meminfo") or ""
        kv = {}
        for line in data.splitlines():
            p = line.split(":")
            if len(p) == 2:
                kv[p[0]] = p[1].strip()

        def kb(k):
            try:
                return int(kv.get(k, "0 kB").split()[0])
            except (ValueError, IndexError):
                return 0
        total = kb("MemTotal")
        avail = kb("MemAvailable")
        used = total - avail
        return {
            "total_mb": round(total / 1024),
            "used_mb": round(used / 1024),
            "percent": round(100.0 * used / total, 1) if total else 0.0,
        }

    # --- misc ------------------------------------------------------------
    def freqs_mhz(self):
        out = []
        base = "/sys/devices/system/cpu"
        try:
            cpus = sorted([d for d in os.listdir(base)
                           if d.startswith("cpu") and d[3:].isdigit()],
                          key=lambda d: int(d[3:]))
        except Exception:
            return out
        for c in cpus:
            raw = _read(f"{base}/{c}/cpufreq/scaling_cur_freq")
            if raw:
                try:
                    out.append(round(int(raw.strip()) / 1000))
                except ValueError:
                    pass
        return out

    def loadavg(self):
        data = (_read("/proc/loadavg") or "").split()
        return [float(x) for x in data[:3]] if len(data) >= 3 else [0, 0, 0]

    def uptime_s(self):
        data = _read("/proc/uptime")
        try:
            return int(float(data.split()[0]))
        except Exception:
            return 0

    @staticmethod
    def fmt_uptime(s):
        d, s = divmod(s, 86400)
        h, s = divmod(s, 3600)
        m, _ = divmod(s, 60)
        if d:
            return f"{d}d {h}h {m}m"
        if h:
            return f"{h}h {m}m"
        return f"{m}m"

    def snapshot(self):
        cpu = self.cpu_percent()
        temps = self.temps()
        cpu_temp = max([v for k, v in temps.items()
                        if "core" in k or "soc" in k] or [0]) or (
            max(temps.values()) if temps else 0)
        return {
            "cpu_total": cpu["total"],
            "cpu_cores": cpu["cores"],
            "cpu_temp": cpu_temp,
            "temps": temps,
            "mem": self.mem(),
            "load": self.loadavg(),
            "freqs_mhz": self.freqs_mhz(),
            "uptime": self.fmt_uptime(self.uptime_s()),
        }


if __name__ == "__main__":
    import json
    si = SysInfo()
    time.sleep(0.3)
    print(json.dumps(si.snapshot(), indent=2))
