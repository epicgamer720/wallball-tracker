#!/usr/bin/env bash
# One-time Pi setup (run with sudo):  sudo bash setup_pi.sh
# - installs Flask + libgpiod (apt; the Pi has no pip)
# - lets the normal user read GPIO without root (gpio group + udev rule)
# - installs the desktop-autostart entry so it launches on boot
set -e

USER_NAME="${SUDO_USER:-darsh}"
HOME_DIR="$(getent passwd "$USER_NAME" | cut -d: -f6)"

echo "[wallball] installing apt deps..."
apt-get update
apt-get install -y python3-flask python3-libgpiod gpiod

echo "[wallball] enabling non-root GPIO for $USER_NAME..."
groupadd -f gpio
usermod -aG gpio "$USER_NAME"
cat >/etc/udev/rules.d/99-gpio.rules <<'EOF'
KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
EOF
udevadm control --reload-rules && udevadm trigger || true

echo "[wallball] installing autostart entry..."
install -d -o "$USER_NAME" -g "$USER_NAME" "$HOME_DIR/.config/autostart"
install -o "$USER_NAME" -g "$USER_NAME" -m 644 \
    "$HOME_DIR/wallball-tracker/wallball.desktop" \
    "$HOME_DIR/.config/autostart/wallball.desktop"
chmod +x "$HOME_DIR/wallball-tracker/run_pi.sh"

echo "[wallball] done."
echo "  -> log out/in (or reboot) so the gpio group + autostart take effect."
echo "  -> wire buttons, then set BTN_PINS in run_pi.sh (find lines with: gpioinfo)"
