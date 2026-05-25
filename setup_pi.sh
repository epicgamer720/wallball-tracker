#!/usr/bin/env bash
# One-shot bootstrap for a fresh Orange Pi (run with sudo):
#   sudo bash ~/wallball-tracker/setup_pi.sh
# Installs all deps, enables non-root GPIO, and installs the boot autostart.
set -e

USER_NAME="${SUDO_USER:-$(id -un)}"
HOME_DIR="$(getent passwd "$USER_NAME" | cut -d: -f6)"

echo "[wallball] installing apt deps (OpenCV, NumPy, Flask, libgpiod, GStreamer tools)..."
apt-get update
apt-get install -y python3-opencv python3-numpy python3-flask \
    python3-libgpiod gpiod gstreamer1.0-tools v4l-utils

echo "[wallball] checking hardware MJPEG decoder (mppjpegdec)..."
if gst-inspect-1.0 mppjpegdec >/dev/null 2>&1; then
    echo "  mppjpegdec present -> hardware decode OK (needed for 1280x720@60)."
else
    echo "  WARNING: mppjpegdec NOT found. High-fps hardware decode will fall back"
    echo "  to CPU V4L2 (~30fps at 1280). It usually ships with the Orange Pi image;"
    echo "  if missing try: sudo apt install -y gstreamer1.0-rockchip1"
fi

echo "[wallball] enabling non-root GPIO for $USER_NAME..."
groupadd -f gpio
usermod -aG gpio "$USER_NAME"
cat >/etc/udev/rules.d/99-gpio.rules <<'EOF'
KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
EOF
udevadm control --reload-rules && udevadm trigger || true

echo "[wallball] installing desktop autostart entry..."
install -d -o "$USER_NAME" -g "$USER_NAME" "$HOME_DIR/.config/autostart"
install -o "$USER_NAME" -g "$USER_NAME" -m 644 \
    "$HOME_DIR/wallball-tracker/wallball.desktop" \
    "$HOME_DIR/.config/autostart/wallball.desktop"
chmod +x "$HOME_DIR/wallball-tracker/run_pi.sh"

echo
echo "[wallball] done. Reboot (or log out/in) so the gpio group + autostart apply."
echo "  Before wiring buttons, verify this board's GPIO chip<->bank mapping:"
echo "    for n in 0 1 2 3 4 5; do echo -n \"gpiochip\$n: \"; readlink -f /sys/bus/gpio/devices/gpiochip\$n; done"
echo "  (gpiochip1 should be fecX0000 = GPIO1, etc. — update BTN_PINS in run_pi.sh if not.)"
