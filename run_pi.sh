#!/usr/bin/env bash
# Launcher + crash supervisor for the wall-ball app (web + on-screen UI).
# Used by the desktop autostart and the "Wall Ball" desktop icon. If webui.py
# ever exits (e.g. an intermittent GStreamer/GLib fault), it relaunches it
# automatically so the dashboard self-heals without a reboot. Logs to wallball.log.
cd "$HOME/wallball-tracker" || exit 1
LOG="$HOME/wallball-tracker/wallball.log"
PIDFILE=/tmp/wallball_run.pid

trap 'exit 0' TERM INT

# Stop a previous supervisor + app instance so a re-launch (boot or icon) is clean.
if [ -f "$PIDFILE" ]; then
    oldpid="$(cat "$PIDFILE" 2>/dev/null)"
    if [ -n "$oldpid" ] && grep -qa run_pi.sh "/proc/$oldpid/cmdline" 2>/dev/null; then
        kill "$oldpid" 2>/dev/null
    fi
fi
pkill -f 'webui[.]py' 2>/dev/null
sleep 1.5
echo $$ > "$PIDFILE"

# Buttons wired to GND (active-low, internal pull-up). libgpiod chip:line,
# order = NEXT, OK, BACK:  GPIO1_A3 -> 1:3   GPIO1_C4 -> 1:20   GPIO2_D3 -> 2:27
BTN_PINS="--btn-pins 1:3,1:20,2:27"

# Supervisor loop: restart the app whenever it exits.
while true; do
    python3 webui.py --screen --wall-side right $BTN_PINS >> "$LOG" 2>&1
    echo "[run_pi] webui exited (code $?) at $(date '+%F %T') — restarting in 3s" >> "$LOG"
    sleep 3
done
