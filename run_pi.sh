#!/usr/bin/env bash
# Launcher used by the desktop-autostart entry. Runs the web UI + the
# on-screen UI on the Pi's monitor. Logs to wallball.log.
cd "$HOME/wallball-tracker" || exit 1

# Stop any instance already running (autostart or a previous click) so we
# never fight over the camera / port 8000.
pkill -f 'webui[.]py' 2>/dev/null && sleep 1.5

# Buttons wired to GND (active-low, internal pull-up). libgpiod chip:line,
# order = NEXT, OK, BACK:
#   GPIO1_A3 -> 1:3    GPIO1_C4 -> 1:20    GPIO2_D3 -> 2:27
BTN_PINS="--btn-pins 1:3,1:20,2:27"

exec python3 webui.py --screen --wall-side right $BTN_PINS \
    >> "$HOME/wallball-tracker/wallball.log" 2>&1
