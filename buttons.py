"""3-button input for the on-screen UI.

Emits three logical events — NEXT, OK, BACK — from either physical GPIO
buttons (via libgpiod) or the keyboard (fallback / no hardware needed).

GPIO wiring (active-low): each button connects its GPIO line to GND, with
the internal pull-up enabled, so an idle line reads 1 and a press reads 0.
Pins are given as "chip:line" (e.g. "4:18"); find valid lines with
`gpioinfo` after `sudo apt install gpiod python3-libgpiod`.

Degrades gracefully: if libgpiod is missing or the lines can't be opened
(e.g. permissions), GPIO is disabled and the keyboard still drives the UI.
"""
import collections
import threading
import time

EVENTS = ("NEXT", "OK", "BACK")


class ButtonInput:
    def __init__(self, pins=None, active_low=True, poll_hz=80, debounce_ms=40):
        self._events = collections.deque()
        self._lock = threading.Lock()
        self._stop = False
        self.active_low = active_low
        self.debounce = debounce_ms / 1000.0
        self.poll_dt = 1.0 / poll_hz
        self.lines = []
        self.gpio_ok = False
        self.gpio_err = "disabled (keyboard only)"
        if pins:
            self._setup_gpio(list(pins)[:3])
        if self.gpio_ok:
            threading.Thread(target=self._loop, daemon=True).start()

    # --- GPIO setup ------------------------------------------------------
    def _setup_gpio(self, pins):
        try:
            import gpiod
        except Exception as e:
            self.gpio_err = f"libgpiod not installed ({e})"
            return
        try:
            for spec in pins:
                chipname, off = self._parse(spec)
                chip = gpiod.Chip(chipname)
                line = chip.get_line(int(off))
                self._request(gpiod, line)
                self.lines.append(line)
            self.gpio_ok = True
            self.gpio_err = ""
        except Exception as e:
            for line in self.lines:
                try:
                    line.release()
                except Exception:
                    pass
            self.lines = []
            self.gpio_ok = False
            self.gpio_err = f"{type(e).__name__}: {e}"

    @staticmethod
    def _request(gpiod, line):
        base = dict(consumer="wallball", type=gpiod.LINE_REQ_DIR_IN)
        try:                       # prefer internal pull-up
            line.request(flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP, **base)
        except Exception:          # older libgpiod: needs an external pull-up
            line.request(**base)

    @staticmethod
    def _parse(spec):
        chip, _, off = str(spec).strip().partition(":")
        if not chip.startswith("gpiochip"):
            chip = f"gpiochip{chip}"
        return chip, off

    # --- polling ---------------------------------------------------------
    def _is_press(self, raw):
        return raw == 0 if self.active_low else raw == 1

    def _loop(self):
        n = len(self.lines)
        stable = [False] * n
        last = [None] * n
        changed_at = [0.0] * n
        while not self._stop:
            now = time.time()
            for i, line in enumerate(self.lines):
                try:
                    pressed = self._is_press(line.get_value())
                except Exception:
                    continue
                if pressed != last[i]:
                    last[i] = pressed
                    changed_at[i] = now
                elif now - changed_at[i] >= self.debounce and pressed != stable[i]:
                    stable[i] = pressed
                    if pressed:                       # falling-edge press
                        self._emit(EVENTS[i])
            time.sleep(self.poll_dt)

    # --- API -------------------------------------------------------------
    def _emit(self, ev):
        with self._lock:
            self._events.append(ev)

    def feed_key(self, ev):
        if ev in EVENTS:
            self._emit(ev)

    def poll(self):
        with self._lock:
            return self._events.popleft() if self._events else None

    def status(self):
        return {"gpio": self.gpio_ok, "detail": self.gpio_err, "lines": len(self.lines)}

    def stop(self):
        self._stop = True
        for line in self.lines:
            try:
                line.release()
            except Exception:
                pass
