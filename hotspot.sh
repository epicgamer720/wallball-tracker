#!/usr/bin/env bash
# Turn the Pi's WiFi into a standalone access point so a phone can reach the
# dashboard with no router / internet (great in the field). Run with sudo:
#     sudo bash ~/wallball-tracker/hotspot.sh [SSID] [PASSWORD]
# Defaults: SSID "WallBall", password "wallball123" (must be >= 8 chars).
#
# After it's up: join that WiFi on your phone, then open  http://10.42.0.1:8000
# Turn it off:        sudo nmcli connection down WallBallAP
# Stop auto-starting: sudo nmcli connection modify WallBallAP connection.autoconnect no
set -e

SSID="${1:-WallBall}"
PASS="${2:-wallball123}"
IFACE="$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="wifi"{print $1; exit}')"
[ -z "$IFACE" ] && { echo "No WiFi device found."; exit 1; }

echo "Setting up access point '$SSID' on $IFACE ..."
nmcli connection delete WallBallAP >/dev/null 2>&1 || true
nmcli connection add type wifi ifname "$IFACE" con-name WallBallAP \
    autoconnect yes ssid "$SSID"
nmcli connection modify WallBallAP \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$PASS"
nmcli connection up WallBallAP

echo
echo "Access point is up and will auto-start on every boot."
echo "  Join WiFi:  $SSID"
echo "  Password:   $PASS"
echo "  Dashboard:  http://10.42.0.1:8000"
echo "(Ethernet keeps working alongside it, so dev/SSH over the LAN is unaffected.)"
