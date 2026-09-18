#!/usr/bin/env bash
#
# Install the Cleo daemon as a systemd service on a Raspberry Pi.
#
#   sudo deploy/install.sh                     # autodetect everything
#   sudo deploy/install.sh --serial FTAAMM58   # pin a specific adapter
#   sudo deploy/install.sh --no-udev           # skip the udev rule
#
# Renders deploy/cleo.service and deploy/99-cleo-servos.rules with the paths of
# this checkout, enables the unit for boot, but does not start it -- starting the
# daemon energises the servos, so that stays a deliberate step.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DEST=/etc/systemd/system/cleo.service
RULES_DEST=/etc/udev/rules.d/99-cleo-servos.rules

SERIAL=""
INSTALL_UDEV=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial) SERIAL="$2"; shift 2 ;;
        --no-udev) INSTALL_UDEV=0; shift ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "error: run with sudo -- this writes to /etc/systemd and /etc/udev" >&2
    exit 1
fi

# --- Who the daemon runs as -------------------------------------------------
# The invoking user, not root: it owns the checkout and the venv, and belongs to
# dialout/video/audio for the adapter, camera and ReSpeaker.
CLEO_USER="${SUDO_USER:-root}"
if [[ "$CLEO_USER" == "root" ]]; then
    echo "warning: no SUDO_USER set, the service will run as root" >&2
fi

# --- The interpreter --------------------------------------------------------
# systemd runs no shell, so ExecStart needs an absolute path. Prefer the uv venv
# and fall back to devenv's, since both are plausible on a Pi.
for candidate in "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/.devenv/state/venv/bin/python3"; do
    if [[ -x "$candidate" ]]; then
        CLEO_PYTHON="$candidate"
        break
    fi
done

if [[ -z "${CLEO_PYTHON:-}" ]]; then
    echo "error: no interpreter found. Run 'uv sync' in $REPO_ROOT first." >&2
    exit 1
fi

echo "repo:        $REPO_ROOT"
echo "user:        $CLEO_USER"
echo "interpreter: $CLEO_PYTHON"

# --- udev rule --------------------------------------------------------------
if [[ $INSTALL_UDEV -eq 1 ]]; then
    # Find the adapter: an explicit --serial wins, otherwise take the only
    # ttyUSB* present. Two of them is ambiguous, so make the user choose.
    if [[ -z "$SERIAL" ]]; then
        mapfile -t tty_candidates < <(ls /dev/ttyUSB* 2>/dev/null || true)
        if [[ ${#tty_candidates[@]} -eq 1 ]]; then
            SERIAL="$(udevadm info -q property -n "${tty_candidates[0]}" |
                      sed -n 's/^ID_SERIAL_SHORT=//p')"
            VENDOR="$(udevadm info -q property -n "${tty_candidates[0]}" |
                      sed -n 's/^ID_VENDOR_ID=//p')"
            echo "adapter:     ${tty_candidates[0]} (serial $SERIAL, vendor $VENDOR)"
        else
            echo "warning: found ${#tty_candidates[@]} ttyUSB devices, cannot autodetect." >&2
            echo "         Re-run with --serial <SERIAL>, or --no-udev to skip." >&2
            INSTALL_UDEV=0
        fi
    else
        VENDOR="${VENDOR:-0403}"  # FTDI, the usual U2D2/FTDI adapter vendor
    fi
fi

if [[ $INSTALL_UDEV -eq 1 && -n "$SERIAL" ]]; then
    sed -e "s|@ID_VENDOR@|${VENDOR:-0403}|" -e "s|@ID_SERIAL@|$SERIAL|" \
        "$REPO_ROOT/deploy/99-cleo-servos.rules" > "$RULES_DEST"
    udevadm control --reload
    udevadm trigger --subsystem-match=tty
    echo "installed:   $RULES_DEST"
else
    echo "skipped:     udev rule (set CLEO_SERIAL in $UNIT_DEST by hand)"
fi

# --- systemd unit -----------------------------------------------------------
sed -e "s|@CLEO_USER@|$CLEO_USER|" \
    -e "s|@CLEO_ROOT@|$REPO_ROOT|" \
    -e "s|@CLEO_PYTHON@|$CLEO_PYTHON|" \
    "$REPO_ROOT/deploy/cleo.service" > "$UNIT_DEST"
echo "installed:   $UNIT_DEST"

systemctl daemon-reload
systemctl enable cleo.service

cat <<EOF

Enabled for boot. Not started -- starting the daemon powers the servos.

  sudo systemctl start cleo     # start now
  systemctl status cleo         # check it came up
  journalctl -u cleo -f         # follow the log
EOF
